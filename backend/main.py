# main.py
import os, uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

# ----------------- FastAPI 기본 설정 -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

# ----------------- 유틸 함수 -----------------
def order_quad(pts: np.ndarray) -> np.ndarray:
    """아무 순서의 4점 -> (TL, TR, BR, BL) 정렬"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]                        # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]                        # TR
    rect[3] = pts[np.argmax(d)]                        # BL
    return rect

def line_at_y(p1, p2, y):
    """점 p1(x1,y1), p2(x2,y2)을 잇는 직선과 수평선 y의 교점 x"""
    x1, y1 = p1; x2, y2 = p2
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) / 2.0
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

def draw_grid_layer(w: int, h: int, rows: int, cols: int,
                    color=(0, 0, 255), thick: int = 4):
    """직사각형 평면에 빨간 격자(1~rows*cols 번호 포함) 그린 이미지 반환"""
    layer = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(layer, (0, 0), (w - 1, h - 1), color, thick)
    for r in range(1, rows):
        y = int(h * r / rows)
        cv2.line(layer, (0, y), (w - 1, y), color, thick)
    for c in range(1, cols):
        x = int(w * c / cols)
        cv2.line(layer, (x, 0), (x, h - 1), color, thick)
    # 칸 번호(1~rows*cols)
    cell = 1
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * w / cols)
            cy = int((r + 0.5) * h / rows)
            cv2.putText(layer, str(cell), (cx - 12, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cell += 1
    return layer

def put_label_center(img, poly_pts, text, color=(255, 0, 0)):
    """폴리곤 중앙(무게중심)에 번호 텍스트 쓰기 (BGR 기본 파란색)"""
    pts = np.array(poly_pts, np.int32)
    M = cv2.moments(pts)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img, str(text), (cx - 18, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4, cv2.LINE_AA)

# ----------------- 핵심 엔드포인트 -----------------
@app.post("/process_video_grid_quad")
async def process_video_grid_quad(
    file: UploadFile = File(...),
    x1: int = Form(...), y1: int = Form(...),
    x2: int = Form(...), y2: int = Form(...),
    x3: int = Form(...), y3: int = Form(...),
    x4: int = Form(...), y4: int = Form(...),
    rows: int = Form(3), cols: int = Form(3),
):
    """
    - 입력: 동영상 + 사다리꼴 꼭짓점 4개 (순서 무관)
    - 처리: (1) 주변 4영역(좌/우/상/하)에 번호 10/11/12/13 표시
            (2) ROI 안에 빨간 3x3 격자(1~9) 오버레이
    - 출력: 결과 이미지(JPEG)
    """

    # 1) 파일 저장 + 중간 프레임 추출
    raw = await file.read()
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.mp4")
    with open(path, "wb") as f:
        f.write(raw)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return JSONResponse({"error": "cannot open video"}, status_code=400)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
    ok, frame = cap.read()
    cap.release()
    # os.remove(path)  # 남기고 싶지 않으면 주석 해제

    if not ok:
        return JSONResponse({"error": "cannot read frame"}, status_code=400)

    H, W = frame.shape[:2]
    src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype="float32")
    src = order_quad(src)  # TL, TR, BR, BL
    tl, tr, br, bl = src

    out = frame.copy()

    # 2) 주변 4영역 폴리곤 계산 (화면 상하단까지 경계 연장)
    Ltop_x    = line_at_y(bl, tl, 0)      # 왼쪽 변, y=0 교점
    Lbottom_x = line_at_y(tl, bl, H - 1)  # 왼쪽 변, y=H-1 교점
    Rtop_x    = line_at_y(br, tr, 0)      # 오른쪽 변, y=0 교점
    Rbottom_x = line_at_y(tr, br, H - 1)  # 오른쪽 변, y=H-1 교점

    top_poly   = np.array([[Ltop_x, 0], [Rtop_x, 0], tr, tl], np.float32)                         # 12
    bot_poly   = np.array([bl, br, [Rbottom_x, H - 1], [Lbottom_x, H - 1]], np.float32)           # 13
    left_poly  = np.array([[0, 0], [Ltop_x, 0], [Lbottom_x, H - 1], [0, H - 1]], np.float32)      # 10
    right_poly = np.array([[Rtop_x, 0], [W - 1, 0], [W - 1, H - 1], [Rbottom_x, H - 1]], np.float32)  # 11

  
    # 3) 번호 + 영역 테두리 표시 (파란색)
    for poly, num in [(left_poly, 10), (right_poly, 11), (top_poly, 12), (bot_poly, 13)]:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 0), thickness=3)  # 테두리
        put_label_center(out, poly, num, color=(255, 0, 0))  # 숫자


    # 4) ROI 안 빨간 격자 오버레이 (원근 변환 사용)
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    dst_w = max(int(max(widthA, widthB)), 100)
    dst_h = max(int(max(heightA, heightB)), 100)

    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
    M  = cv2.getPerspectiveTransform(src, dst)
    iM = np.linalg.inv(M)

    grid = draw_grid_layer(dst_w, dst_h, rows, cols, color=(0, 0, 255), thick=4)  # 빨강
    grid_back = cv2.warpPerspective(grid, iM, (W, H), flags=cv2.INTER_LINEAR)

    mask = cv2.cvtColor(grid_back, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)[:, :, None]
    out = np.where(mask == 1, grid_back, out)

    # 5) JPEG로 응답
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return JSONResponse({"error": "encode failed"}, status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
