# main.py (FastAPI)
import os, uuid
import numpy as np, cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
UPLOAD_DIR = "uploads"; os.makedirs(UPLOAD_DIR, exist_ok=True)

def order_quad(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]                        # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]                        # TR
    rect[3] = pts[np.argmax(d)]                        # BL
    return rect

def line_at_y(p1, p2, y):
    x1, y1 = p1; x2, y2 = p2
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) / 2.0
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

def draw_grid_layer(w: int, h: int, rows: int, cols: int,
                    color=(0, 0, 255), thick: int = 4):
    layer = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(layer, (0, 0), (w - 1, h - 1), color, thick)
    for r in range(1, rows):
        y = int(h * r / rows); cv2.line(layer, (0, y), (w - 1, y), color, thick)
    for c in range(1, cols):
        x = int(w * c / cols); cv2.line(layer, (x, 0), (x, h - 1), color, thick)
    # 번호(1~rows*cols)
    n = 1
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * w / cols); cy = int((r + 0.5) * h / rows)
            cv2.putText(layer, str(n), (cx - 12, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            n += 1
    return layer

def put_label_center(img, poly_pts, text, color=(255, 0, 0)):
    pts = np.array(poly_pts, np.int32)
    M = cv2.moments(pts)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        cv2.putText(img, str(text), (cx - 18, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4, cv2.LINE_AA)

@app.post("/make_overlay_quad")
async def make_overlay_quad(
    file: UploadFile = File(...),
    x1: int = Form(...), y1: int = Form(...),
    x2: int = Form(...), y2: int = Form(...),
    x3: int = Form(...), y3: int = Form(...),
    x4: int = Form(...), y4: int = Form(...),
    rows: int = Form(3), cols: int = Form(3),
):
    # 1) 비디오 저장 & 해상도 파악용 첫/중간 프레임 추출
    raw = await file.read()
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.mp4")
    with open(path, "wb") as f: f.write(raw)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return JSONResponse({"error":"cannot open"}, 400)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total//2, 0))
    ok, frame = cap.read()
    cap.release()
    # os.remove(path)  # 필요시 삭제
    if not ok: return JSONResponse({"error":"cannot read frame"}, 400)

    H, W = frame.shape[:2]
    src = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype="float32")
    src = order_quad(src)
    tl, tr, br, bl = src

    # --- 바깥 10/11/12/13 영역 폴리곤 계산 ---
    Ltop_x    = line_at_y(bl, tl, 0)
    Lbottom_x = line_at_y(tl, bl, H-1)
    Rtop_x    = line_at_y(br, tr, 0)
    Rbottom_x = line_at_y(tr, br, H-1)

    top_poly   = np.array([[Ltop_x,0],[Rtop_x,0],tr,tl], np.float32)                    # 12
    bot_poly   = np.array([bl,br,[Rbottom_x,H-1],[Lbottom_x,H-1]], np.float32)          # 13
    left_poly  = np.array([[0,0],[Ltop_x,0],[Lbottom_x,H-1],[0,H-1]], np.float32)       # 10
    right_poly = np.array([[Rtop_x,0],[W-1,0],[W-1,H-1],[Rbottom_x,H-1]], np.float32)   # 11

    # --- 격자(warp) 생성 (BGR) ---
    widthA  = np.linalg.norm(br - bl); widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
    dst_w = max(int(max(widthA, widthB)), 100)
    dst_h = max(int(max(heightA, heightB)), 100)
    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype="float32")
    M  = cv2.getPerspectiveTransform(src, dst)
    iM = np.linalg.inv(M)

    grid = draw_grid_layer(dst_w, dst_h, rows, cols, color=(0,0,255), thick=4)  # 빨강
    grid_back = cv2.warpPerspective(grid, iM, (W, H), flags=cv2.INTER_LINEAR)

    # --- 오버레이 캔버스 만들고: 바깥 영역 테두리 + 번호, 그리고 격자 선 합성 ---
    overlay = np.zeros((H, W, 3), np.uint8)

    # 바깥 영역 테두리 + 번호(파란색)
    for poly, num in [(left_poly,10), (right_poly,11), (top_poly,12), (bot_poly,13)]:
        pts = np.array(poly, np.int32).reshape((-1,1,2))
        cv2.polylines(overlay, [pts], True, (255,0,0), 3)
        put_label_center(overlay, poly, num, color=(255,0,0))

    # 격자(빨강) 선을 overlay에 복사(비영점 픽셀만)
    mask_grid = cv2.cvtColor(grid_back, cv2.COLOR_BGR2GRAY)
    overlay[mask_grid > 0] = grid_back[mask_grid > 0]

    # --- 투명 PNG(BGRA)로 변환: 선/텍스트 부분만 알파=255, 나머지=0 ---
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    alpha = np.where(gray > 0, 255, 0).astype(np.uint8)
    bgra = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    ok, buf = cv2.imencode(".png", bgra)  # PNG(투명)
    if not ok: return JSONResponse({"error":"encode failed"}, 500)
    return Response(content=buf.tobytes(), media_type="image/png")
