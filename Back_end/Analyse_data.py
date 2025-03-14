import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import json
import os

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时存储上传文件
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "saved_path": file_path
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"文件上传失败: {str(e)}"}
        )

@app.get("/api/statistics")
async def get_statistics(filename: str):
    try:
        df = pd.read_excel(os.path.join(UPLOAD_DIR, filename))
        stats = df.describe().to_dict()
        
        return JSONResponse({
            "data_stats": stats,
            "columns": list(df.columns),
            "sample_data": df.head(5).to_dict(orient="records")
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"数据分析失败: {str(e)}"}
        )

@app.get("/api/filter")
async def filter_data(template_filename: str, data_filename: str):
    try:
        # 读取模板文件获取列名
        template_path = os.path.join(UPLOAD_DIR, template_filename)
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail="模板文件不存在")
            
        template_df = pd.read_excel(template_path)
        required_columns = template_df.columns.tolist()

        # 读取全量数据文件
        data_path = os.path.join(UPLOAD_DIR, data_filename)
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="数据文件不存在")
            
        full_df = pd.read_excel(data_path)

        # 列名校验
        missing_columns = [col for col in required_columns if col not in full_df.columns]
        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={"error": "以下列在数据文件中不存在", "missing_columns": missing_columns}
            )

        # 筛选数据
        filtered_df = full_df[required_columns]

        return JSONResponse({
            "filtered_data": filtered_df.head(100).to_dict(orient="records"),
            "available_columns": required_columns,
            "total_rows": len(filtered_df)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"数据过滤失败: {str(e)}"}
        )


@app.get("/api/visualization")
async def get_visualization_data(request: Request, filename: str):
    try:
        df = pd.read_excel(os.path.join(UPLOAD_DIR, filename))
        
        # 获取请求参数
        x_col = request.query_params.get('x_col', df.columns[0])
        y_col = request.query_params.get('y_col', df.columns[3])
        
        # 校验列是否存在
        if x_col not in df.columns or y_col not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": "选择的列不存在于数据集中"}
            )
        
        # 时间序列数据转换（保留原有逻辑但不再固定列名）
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = pd.to_datetime(df[col])
        
        # 趋势图数据
        trend_data = {
            "x": df[x_col].astype(str).tolist(),
            "y": df[y_col].astype(str).tolist()
        }
        
        # 散点图数据
        scatter_data = {
            "x": df[x_col].tolist(),
            "y": df[y_col].tolist()
        }
        
        return JSONResponse({
            "trend_data": trend_data,
            "scatter_data": scatter_data,
            "available_columns": list(df.columns)
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"可视化数据处理失败: {str(e)}"}
        )
