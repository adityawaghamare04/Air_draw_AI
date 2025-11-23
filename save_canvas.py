# save_canvas.py
import numpy as np
import cv2

points = np.load("draw_points.npy", allow_pickle=True)

# Convert to numpy array (N, 2)
points = np.array(points, dtype=np.float32)
xs, ys = points[:, 0], points[:, 1]

# Normalize into [0, 255] range
canvas_size = 256
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

# Avoid division by zero
range_x = max(max_x - min_x, 1)
range_y = max(max_y - min_y, 1)

norm_xs = (xs - min_x) / range_x * (canvas_size - 1)
norm_ys = (ys - min_y) / range_y * (canvas_size - 1)

canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

for x, y in zip(norm_xs, norm_ys):
    cv2.circle(canvas, (int(x), int(y)), 2, 255, -1)

# Optionally resize to 28x28 here:
small = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

cv2.imwrite("air_drawing.png", small)
cv2.imshow("Drawing", small)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Saved air_drawing.png")
