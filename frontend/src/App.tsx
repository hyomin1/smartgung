import { useEffect, useRef, useState } from 'react';

type Pt = { x: number; y: number };

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [pts, setPts] = useState<Pt[]>([]);
  const [overlayUrl, setOverlayUrl] = useState<string>('');
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setPts([]);
    setOverlayUrl('');
    if (inputRef.current) inputRef.current.value = '';
  };

  useEffect(() => {
    if (!file) {
      setVideoUrl('');
      return;
    }
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const addPoint = (e: React.MouseEvent) => {
    if (!overlayRef.current || pts.length >= 4) return;
    const rect = overlayRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
    setPts((p) => [...p, { x, y }]);
  };

  const makeOverlay = async () => {
    if (!file || !videoRef.current || pts.length !== 4) return;
    const v = videoRef.current;

    // 영상 일시정지 상태에서 점 찍기 추천
    v.pause();

    // 표시크기 -> 원본 해상도 스케일 보정
    const dispW = v.clientWidth,
      dispH = v.clientHeight;
    const natW = v.videoWidth || dispW,
      natH = v.videoHeight || dispH;
    const sx = natW / dispW,
      sy = natH / dispH;
    const S = pts.map((p) => ({
      x: Math.round(p.x * sx),
      y: Math.round(p.y * sy),
    }));

    const fd = new FormData();
    fd.append('file', file, file.name);
    fd.append('x1', String(S[0].x));
    fd.append('y1', String(S[0].y));
    fd.append('x2', String(S[1].x));
    fd.append('y2', String(S[1].y));
    fd.append('x3', String(S[2].x));
    fd.append('y3', String(S[2].y));
    fd.append('x4', String(S[3].x));
    fd.append('y4', String(S[3].y));
    fd.append('rows', '3');
    fd.append('cols', '3');

    const res = await fetch('http://localhost:8000/make_overlay_quad', {
      method: 'POST',
      body: fd,
    });
    if (!res.ok) {
      alert('서버 오류');
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    setOverlayUrl(url);

    // 이제 재생
    v.play().catch(() => {});
  };

  return (
    <div style={{ padding: 24, display: 'grid', gap: 12 }}>
      <input ref={inputRef} type='file' accept='video/*' onChange={onPick} />

      {videoUrl && (
        <div style={{ position: 'relative', width: 640, maxWidth: '95vw' }}>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            style={{ width: '100%', display: 'block' }}
            muted
            playsInline
          />
          {/* 점 4개 찍기 오버레이 (오버레이 PNG 생기면 숨겨도 됨) */}
          {!overlayUrl && (
            <div
              ref={overlayRef}
              onClick={addPoint}
              style={{ position: 'absolute', inset: 0, cursor: 'crosshair' }}
              title='사다리꼴 4점을 찍어주세요'
            />
          )}
          {/* 점/선 미리보기 */}
          {!overlayUrl &&
            pts.map((p, i) => (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: p.x - 6,
                  top: p.y - 6,
                  width: 12,
                  height: 12,
                  borderRadius: 12,
                  background: 'red',
                  pointerEvents: 'none',
                }}
              />
            ))}
          {!overlayUrl && pts.length >= 2 && (
            <svg
              style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
              viewBox={`0 0 ${overlayRef.current?.clientWidth ?? 0} ${
                overlayRef.current?.clientHeight ?? 0
              }`}
              preserveAspectRatio='none'
            >
              <polyline
                points={pts.map((p) => `${p.x},${p.y}`).join(' ')}
                fill='none'
                stroke='red'
                strokeWidth={3}
              />
              {pts.length === 4 && (
                <line
                  x1={pts[3].x}
                  y1={pts[3].y}
                  x2={pts[0].x}
                  y2={pts[0].y}
                  stroke='red'
                  strokeWidth={3}
                />
              )}
            </svg>
          )}

          {/* 서버가 준 투명 PNG 오버레이: 영상 위에 겹침 */}
          {overlayUrl && (
            <img
              src={overlayUrl}
              alt='overlay'
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={() => setPts([])}
          disabled={pts.length === 0 || !!overlayUrl}
        >
          점 초기화
        </button>
        <button
          onClick={makeOverlay}
          disabled={!file || pts.length !== 4 || !!overlayUrl}
        >
          오버레이 생성 후 재생
        </button>
      </div>
    </div>
  );
}
