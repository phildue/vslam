# Copyright 2022 Philipp.Duernay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import cv2 as cv


def estimate_E_compute_F(pts1, pts2, K):
    E, mask1 = cv.findEssentialMat(pts1, pts2, K, method=cv.FM_LMEDS)
    retval, R, t, mask2 = cv.recoverPose(E, pts1[mask1.ravel() == 1], pts2[mask1.ravel() == 1], K)
    tcross = np.cross(t.reshape((3,)), np.identity(3)*-1)
    E = tcross.dot(R)
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv
    return F, mask1, R, t


def extract_matches(img1, img2):
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda val: val[1].distance)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def triangulate_dlt(uv1, uv2, P1, P2):
    """Triangulation with direct linear transform (DLT)."""
    n_points = uv1.shape[1]
    p3d = np.zeros((3, n_points))
    for i in range(0, n_points):
        A = np.zeros((4, 4))  # (n views, 4)
        u1 = uv1[0, i]
        v1 = uv1[1, i]
        u2 = uv2[0, i]
        v2 = uv2[1, i]
        A[0] = u1*P1[2] - P1[0]
        A[1] = v1*P1[2] - P1[1]
        A[2] = u2*P2[2] - P2[0]
        A[3] = v2*P2[2] - P2[1]
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        p3d[:, i] = (vh[-1]/vh[-1, -1])[:3]
    """
    This should also work no?
    A = np.hstack(As)
    print(A.shape)
    u, s, vh = np.linalg.svd(A,full_matrices=True)
    print(vh.shape)
    p3d = vh[-1].reshape((4,n_points))
    p3d /= p3d[-1]
    """
    return p3d[:3]


if __name__ == "__main__":
    LOG = True
    folder = '/media/data/dataset/rgbd_dataset_freiburg2_desk/rgbd_dataset_freiburg2_desk/rgb'
    img1 = cv.imread(f'{folder}/1311868164.399026.png', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(f'{folder}/1311868165.599188.png', cv.IMREAD_GRAYSCALE)
    K = np.array([[525.0, 0, 319.5],
                  [0, 525.0, 239.5],
                  [0, 0, 1]])

    pts1, pts2 = extract_matches(img1, img2)
    F, mask, R, t = estimate_E_compute_F(pts1, pts2, K)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # uv1 = K (zR(K'uv0)+t)
    # x1'Fx0 = 0
    x2 = np.row_stack([pts2[:, 0], pts2[:, 1], np.ones((pts2.shape[0],))])
    x1 = np.row_stack([pts1[:, 0], pts1[:, 1], np.ones((pts1.shape[0],))])
    lines1 = F.T @ x2
    error = (lines1.T * x1.T).sum(-1)  # equivalent of diag(lines1.T @ x1)
    print(f"error = {error.mean():.2f} +- {error.std():.2f}")
    lines2 = F @ x1

    Rt = np.identity(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.ravel()
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))]) @ np.eye(4)
    P2 = K @ np.hstack([np.eye(3), np.zeros((3, 1))]) @ Rt

    p3d = triangulate_dlt(x1, x2, P1, P2)
    pts1_ = P1 @ np.vstack([p3d, np.ones((p3d.shape[1],))])
    pts1_ /= pts1_[2]
    error1 = np.linalg.norm(pts1_[:2].T - pts1)

    pts2_ = P2 @ np.vstack([p3d, np.ones((p3d.shape[1],))])
    pts2_ /= pts2_[2]
    error2 = np.linalg.norm(pts2_[:2].T - pts2)
    print(f"Error: {error1+error2}")

    print(f"K={K}")
    print(f"Rt={Rt}")
    print(f"P1={P1}")
    print(f"P2={P2}")
    print(f"p3d   = {p3d.T[:10]}")
    print(f"pts1_ = {pts1_.T[:10]}")

    img5, img6 = drawlines(img1, img2, lines1.T, pts1, pts2)
    img3, img4 = drawlines(img2, img1, lines2.T, pts2, pts1)
    img_stack = np.hstack([img5, img3])
    if LOG:
        np.savetxt('points3d.csv', p3d.T, delimiter=',', fmt='%.3f')
        np.savetxt('observations1.csv', pts1, delimiter=',', fmt='%.3f')
        np.savetxt('observations2.csv', pts2, delimiter=',', fmt='%.3f')
        np.savetxt('Rt.csv', Rt, delimiter=',')
        cv.imshow("Out", img_stack)
        cv.waitKey(0)
