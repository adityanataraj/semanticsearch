import io
import csv
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Semantic Search for CSV")

# Global state for simplicity (single-user tool)
model: Optional[SentenceTransformer] = None
datasets: dict = {}  # dataset_id -> {df, embeddings, text_column, columns}

UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def get_model() -> SentenceTransformer:
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    contents = await file.read()
    if len(contents) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

    try:
        # Try common encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = contents.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Could not decode file. Supported encodings: UTF-8, Latin-1, CP1252")

        # Detect delimiter
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except csv.Error:
            sep = ","

        df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Identify text columns (object dtype with reasonable text content)
    text_columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 5:  # Skip columns that are just short codes
                text_columns.append(col)

    if not text_columns:
        # Fall back to all object columns
        text_columns = [col for col in df.columns if df[col].dtype == "object"]

    if not text_columns:
        raise HTTPException(status_code=400, detail="No text columns found in CSV")

    # Combine text columns for embedding
    df["_combined_text"] = df[text_columns].fillna("").astype(str).agg(" | ".join, axis=1)

    # Generate embeddings
    m = get_model()
    texts = df["_combined_text"].tolist()
    embeddings = m.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dataset_id = str(uuid.uuid4())[:8]

    # Detect column types for filter UI
    column_info = []
    for col in df.columns:
        if col == "_combined_text":
            continue
        col_type = "text"
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "date"
        else:
            # Try to parse as date
            try:
                pd.to_datetime(df[col], format="mixed", dayfirst=False)
                col_type = "date"
            except (ValueError, TypeError):
                pass

        unique_vals = None
        if col_type == "text" and df[col].nunique() <= 50:
            unique_vals = sorted(df[col].dropna().unique().astype(str).tolist())

        column_info.append({
            "name": col,
            "type": col_type,
            "unique_values": unique_vals,
        })

    datasets[dataset_id] = {
        "df": df,
        "embeddings": embeddings,
        "text_columns": text_columns,
        "columns": column_info,
        "filename": file.filename,
    }

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": column_info,
        "text_columns": text_columns,
    }


@app.get("/api/search")
async def search(
    dataset_id: str = Query(...),
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    columns: Optional[str] = Query(None, description="Comma-separated column names to search in"),
    filters: Optional[str] = Query(None, description="JSON-encoded filters"),
):
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload a CSV first.")

    data = datasets[dataset_id]
    df = data["df"]
    embeddings = data["embeddings"]

    m = get_model()
    query_embedding = m.encode([q], show_progress_bar=False)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # If specific columns requested, re-compute similarity against those columns only
    if columns:
        search_cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if search_cols:
            combined = df[search_cols].fillna("").astype(str).agg(" | ".join, axis=1)
            col_embeddings = m.encode(combined.tolist(), show_progress_bar=False, batch_size=64)
            col_embeddings = col_embeddings / np.linalg.norm(col_embeddings, axis=1, keepdims=True)
            similarities = np.dot(col_embeddings, query_embedding.T).flatten()
        else:
            similarities = np.dot(embeddings, query_embedding.T).flatten()
    else:
        similarities = np.dot(embeddings, query_embedding.T).flatten()

    # Apply filters
    mask = np.ones(len(df), dtype=bool)
    if filters:
        import json
        try:
            filter_list = json.loads(filters)
        except json.JSONDecodeError:
            filter_list = []

        for f in filter_list:
            col = f.get("column")
            op = f.get("op", "eq")
            val = f.get("value")
            if col not in df.columns or val is None:
                continue

            if op == "eq":
                mask &= df[col].astype(str) == str(val)
            elif op == "neq":
                mask &= df[col].astype(str) != str(val)
            elif op == "contains":
                mask &= df[col].astype(str).str.contains(str(val), case=False, na=False)
            elif op == "gt":
                try:
                    mask &= pd.to_numeric(df[col], errors="coerce") > float(val)
                except (ValueError, TypeError):
                    pass
            elif op == "lt":
                try:
                    mask &= pd.to_numeric(df[col], errors="coerce") < float(val)
                except (ValueError, TypeError):
                    pass

    # Apply mask to similarities (filtered-out rows get -1)
    filtered_similarities = similarities.copy()
    filtered_similarities[~mask] = -1.0

    # Rank by similarity
    ranked_indices = np.argsort(-filtered_similarities)
    top_indices = ranked_indices[offset:offset + limit]

    # Build results
    display_cols = [c for c in df.columns if c != "_combined_text"]
    results = []
    for idx in top_indices:
        score = float(filtered_similarities[idx])
        if score < 0:
            break
        row = df.iloc[idx]
        row_data = {}
        for col in display_cols:
            val = row[col]
            if pd.isna(val):
                row_data[col] = None
            else:
                row_data[col] = str(val)
        results.append({
            "index": int(idx),
            "score": round(score * 100, 1),
            "data": row_data,
        })

    total_matching = int(mask.sum())
    return {
        "query": q,
        "total_matching": total_matching,
        "returned": len(results),
        "offset": offset,
        "results": results,
    }


@app.get("/api/export")
async def export_results(
    dataset_id: str = Query(...),
    q: str = Query(..., min_length=1),
    limit: int = Query(500, ge=1, le=5000),
    columns: Optional[str] = Query(None),
    filters: Optional[str] = Query(None),
):
    """Export search results back to CSV."""
    search_result = await search(
        dataset_id=dataset_id, q=q, limit=limit, offset=0,
        columns=columns, filters=filters,
    )

    if not search_result["results"]:
        raise HTTPException(status_code=404, detail="No results to export")

    rows = []
    for r in search_result["results"]:
        row = r["data"].copy()
        row["_similarity_score"] = r["score"]
        rows.append(row)

    output_df = pd.DataFrame(rows)
    buffer = io.StringIO()
    output_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=search_results.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
