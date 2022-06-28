# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time


LX = 2
LY = 2
LZ = 2
geneNum = 500
Nair = 1  # 空気の屈折率

rayStartV = np.array([10000*100, 0*100, 0*100])  # m から cm へ変換
centerX = 0  # 入射光表示の中心座標
centerY = 0  # 入射光表示の中心座標
centerZ = 0  # 入射光表示の中心座標
rayDensity = 0.25  # 入射光の密度
focusX = 10.7  # 焦点付近の描画範囲を平行移動

screenV = np.array([10.86973606, 0, 0])  # スクリーンの位置ベクトル
UnitX = -0


lens1V = np.array([-4.456+UnitX, 0, 0])  # レンズ１の位置ベクトル
lens2V = np.array([-3.7+UnitX, 0, 0])  # レンズ２の位置ベクトル
lens3V = np.array([-2.57+UnitX, 0, 0])  # レンズ３の位置ベクトル
lens4V = np.array([-0.78+UnitX, 0, 0])  # レンズ４の位置ベクトル
lens5V = np.array([-0.58+UnitX, 0, 0])  # レンズ5の位置ベクトル

Lens1Param = [4.78, 37.2/4, 0.8-0.044, [1.791, 1.791], [-1, 1], lens1V]
Lens2Param = [37.2/4, 26.25, 0.27, [1.788, 1.788], [1, -1], lens2V]
Lens3Param = [11.27, 4.3, 0.514, [1.26, 1.26], [1, -1], lens3V]
Lens4Param = [1000000, 26.32/4, 0.2, [1.395, 1.395], [1, -1], lens4V]
Lens5Param = [26.32/4, 9.305, 0.58, [1.395, 1.395], [-1, 1], lens5V]

# レンズ曲率半径
Rx11 = Lens1Param[0]
Rx12 = Lens1Param[1]
Rx21 = Lens2Param[0]
Rx22 = Lens2Param[1]
Rx31 = Lens3Param[0]
Rx32 = Lens3Param[1]
Rx41 = Lens4Param[0]
Rx42 = Lens4Param[1]
Rx51 = Lens5Param[0]
Rx52 = Lens5Param[1]

# レンズ厚さ
Lensd1 = Lens1Param[2]
Lensd2 = Lens2Param[2]
Lensd3 = Lens3Param[2]
Lensd4 = Lens4Param[2]
Lensd5 = Lens5Param[2]

Params = [Lens1Param, Lens2Param, Lens3Param, Lens4Param, Lens5Param]

# 実測値と比較
paramList = [
    [1.60008, 1.00622],  # N-SK14  確定
    [1.49552, 1.00497],  # N-BK10  確定
    [1.61506, 1.01054],  # N-F2    確定
    [1.50592, 1.00552],  # N-ZK7   確定
    [1.58619, 1.00606]   # N-SK5   確定
]

Nlens1 = paramList[0][0]
NBlueRay1 = paramList[0][1]
Nlens2 = paramList[1][0]
NBlueRay2 = paramList[1][1]
Nlens3 = paramList[2][0]
NBlueRay3 = paramList[2][1]
Nlens4 = paramList[3][0]
NBlueRay4 = paramList[3][1]
Nlens5 = paramList[4][0]
NBlueRay5 = paramList[4][1]


class VectorFunctions:
    # 受け取ったx,y,z座標から(x,y,z)の組を作る関数
    def makePoints(self, point0, point1, point2, shape0, shape1):
        result = [None]*(len(point0)+len(point1)+len(point2))
        result[::3] = point0
        result[1::3] = point1
        result[2::3] = point2
        result = np.array(result)
        result = result.reshape(shape0, shape1)
        return result

    # レイトレーシング、光線ベクトルとレンズ１の交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Lens1L(self, startV, directionV):
        startV = startV - lens1V
        A = (directionV[0]**2/Rx11**2)+(
            directionV[1]**2/Rx11**2)+(
            directionV[2]**2/Rx11**2)
        # print(A)
        B = ((startV[0] - Rx11)*directionV[0]/Rx11**2)+(
            startV[1]*directionV[1]/Rx11**2)+(
            startV[2]*directionV[2]/Rx11**2)
        # print(B)
        C = -1+((startV[0] - Rx11)**2/Rx11**2)+(
            startV[1]**2/Rx11**2)+(
            startV[2]**2/Rx11**2)
        # print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens1R(self, startV, directionV):
        startV = startV - lens1V
        A = (directionV[0]**2/Rx12**2)+(
            directionV[1]**2/Rx12**2)+(
            directionV[2]**2/Rx12**2)
        # print(A)
        B = ((startV[0] + Rx12 - Lensd1)*directionV[0]/Rx12**2)+(
            startV[1]*directionV[1]/Rx12**2)+(
            startV[2]*directionV[2]/Rx12**2)
        # print(B)
        C = -1+((startV[0] + Rx12 - Lensd1)**2/Rx12**2)+(
            startV[1]**2/Rx12**2)+(
            startV[2]**2/Rx12**2)
        # print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ１表面の法線を求める関数
    def decideNormalV_Lens1L(self, pointV):
        pointV = pointV - lens1V
        nornalVx = (2/Rx11**2)*(pointV[0] - Rx11)
        nornalVy = (2/Rx11**2)*pointV[1]
        nornalVz = (2/Rx11**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens1R(self, pointV):
        pointV = pointV - lens1V
        nornalVx = (2/Rx12**2)*(pointV[0] + Rx12 - Lensd1)
        nornalVy = (2/Rx12**2)*pointV[1]
        nornalVz = (2/Rx12**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    # レイトレーシング、光線ベクトルとレンズ２の交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Lens2L(self, startV, directionV):
        startV = startV - lens2V
        A = (directionV[0]**2/Rx21**2)+(
            directionV[1]**2/Rx21**2)+(
            directionV[2]**2/Rx21**2)
        # print(A)
        B = ((startV[0] + Rx21)*directionV[0]/Rx21**2)+(
            startV[1]*directionV[1]/Rx21**2)+(
            startV[2]*directionV[2]/Rx21**2)
        # print(B)
        C = -1+((startV[0] + Rx21)**2/Rx21**2)+(
            startV[1]**2/Rx21**2)+(
            startV[2]**2/Rx21**2)
        # print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens2R(self, startV, directionV):
        startV = startV - lens2V
        A = (directionV[0]**2/Rx22**2)+(
            directionV[1]**2/Rx22**2)+(
            directionV[2]**2/Rx22**2)
        # print(A)
        B = ((startV[0] - Rx22 - Lensd2)*directionV[0]/Rx22**2)+(
            startV[1]*directionV[1]/Rx22**2)+(
            startV[2]*directionV[2]/Rx22**2)
        # print(B)
        C = -1+((startV[0] - Rx22 - Lensd2)**2/Rx22**2)+(
            startV[1]**2/Rx22**2)+(
            startV[2]**2/Rx22**2)
        # print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ２表面の法線を求める関数
    def decideNormalV_Lens2L(self, pointV):
        pointV = pointV - lens2V
        nornalVx = -(2/Rx21**2)*(pointV[0] + Rx21)
        nornalVy = -(2/Rx21**2)*pointV[1]
        nornalVz = -(2/Rx21**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens2R(self, pointV):
        pointV = pointV - lens2V
        nornalVx = -(2/Rx22**2)*(pointV[0] - Rx22 - Lensd2)
        nornalVy = -(2/Rx22**2)*pointV[1]
        nornalVz = -(2/Rx22**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    # レイトレーシング、光線ベクトルとレンズ３の交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Lens3L(self, startV, directionV):
        startV = startV - lens3V
        A = (directionV[0]**2/Rx31**2)+(
            directionV[1]**2/Rx31**2)+(
            directionV[2]**2/Rx31**2)
        # print(A)
        B = ((startV[0] + Rx31)*directionV[0]/Rx31**2)+(
            startV[1]*directionV[1]/Rx31**2)+(
            startV[2]*directionV[2]/Rx31**2)
        # print(B)
        C = -1+((startV[0] + Rx31)**2/Rx31**2)+(
            startV[1]**2/Rx31**2)+(
            startV[2]**2/Rx31**2)
        # print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens3R(self, startV, directionV):
        startV = startV - lens3V
        A = (directionV[0]**2/Rx32**2)+(
            directionV[1]**2/Rx32**2)+(
            directionV[2]**2/Rx32**2)
        # print(A)
        B = ((startV[0] - Rx32 - Lensd3)*directionV[0]/Rx32**2)+(
            startV[1]*directionV[1]/Rx32**2)+(
            startV[2]*directionV[2]/Rx32**2)
        # print(B)
        C = -1+((startV[0] - Rx32 - Lensd3)**2/Rx32**2)+(
            startV[1]**2/Rx32**2)+(
            startV[2]**2/Rx32**2)
        # print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ３表面の法線を求める関数
    def decideNormalV_Lens3L(self, pointV):
        pointV = pointV - lens3V
        nornalVx = -(2/Rx31**2)*(pointV[0] + Rx31)
        nornalVy = -(2/Rx31**2)*pointV[1]
        nornalVz = -(2/Rx31**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens3R(self, pointV):
        pointV = pointV - lens3V
        nornalVx = -(2/Rx32**2)*(pointV[0] - Rx32 - Lensd3)
        nornalVy = -(2/Rx32**2)*pointV[1]
        nornalVz = -(2/Rx32**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    # レイトレーシング、光線ベクトルとレンズ４の交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Lens4L(self, startV, directionV):
        startV = startV - lens4V
        A = (directionV[0]**2/Rx41**2)+(
            directionV[1]**2/Rx41**2)+(
            directionV[2]**2/Rx41**2)
        # print(A)
        B = ((startV[0] + Rx41)*directionV[0]/Rx41**2)+(
            startV[1]*directionV[1]/Rx41**2)+(
            startV[2]*directionV[2]/Rx41**2)
        # print(B)
        C = -1+((startV[0] + Rx41)**2/Rx41**2)+(
            startV[1]**2/Rx41**2)+(
            startV[2]**2/Rx41**2)
        # print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens4R(self, startV, directionV):
        startV = startV - lens4V
        A = (directionV[0]**2/Rx42**2)+(
            directionV[1]**2/Rx42**2)+(
            directionV[2]**2/Rx42**2)
        # print(A)
        B = ((startV[0] - Rx42 - Lensd4)*directionV[0]/Rx42**2)+(
            startV[1]*directionV[1]/Rx42**2)+(
            startV[2]*directionV[2]/Rx42**2)
        # print(B)
        C = -1+((startV[0] - Rx42 - Lensd4)**2/Rx42**2)+(
            startV[1]**2/Rx42**2)+(
            startV[2]**2/Rx42**2)
        # print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ４表面の法線を求める関数
    def decideNormalV_Lens4L(self, pointV):
        pointV = pointV - lens4V
        nornalVx = -(2/Rx41**2)*(pointV[0] + Rx41)
        nornalVy = -(2/Rx41**2)*pointV[1]
        nornalVz = -(2/Rx41**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens4R(self, pointV):
        pointV = pointV - lens4V
        nornalVx = -(2/Rx42**2)*(pointV[0] - Rx42 - Lensd4)
        nornalVy = -(2/Rx42**2)*pointV[1]
        nornalVz = -(2/Rx42**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    # レイトレーシング、光線ベクトルとレンズ5の交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Lens5L(self, startV, directionV):
        startV = startV - lens5V
        A = (directionV[0]**2/Rx51**2)+(
            directionV[1]**2/Rx51**2)+(
            directionV[2]**2/Rx51**2)
        # print(A)
        B = ((startV[0] - Rx51)*directionV[0]/Rx51**2)+(
            startV[1]*directionV[1]/Rx51**2)+(
            startV[2]*directionV[2]/Rx51**2)
        # print(B)
        C = -1+((startV[0] - Rx51)**2/Rx51**2)+(
            startV[1]**2/Rx51**2)+(
            startV[2]**2/Rx51**2)
        # print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens5R(self, startV, directionV):
        startV = startV - lens5V
        A = (directionV[0]**2/Rx52**2)+(
            directionV[1]**2/Rx52**2)+(
            directionV[2]**2/Rx52**2)
        # print(A)
        B = ((startV[0] + Rx52 - Lensd5)*directionV[0]/Rx52**2)+(
            startV[1]*directionV[1]/Rx52**2)+(
            startV[2]*directionV[2]/Rx52**2)
        # print(B)
        C = -1+((startV[0] + Rx52 - Lensd5)**2/Rx52**2)+(
            startV[1]**2/Rx52**2)+(
            startV[2]**2/Rx52**2)
        # print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ5表面の法線を求める関数
    def decideNormalV_Lens5L(self, pointV):
        pointV = pointV - lens5V
        nornalVx = (2/Rx51**2)*(pointV[0] - Rx51)
        nornalVy = (2/Rx51**2)*pointV[1]
        nornalVz = (2/Rx51**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens5R(self, pointV):
        pointV = pointV - lens5V
        nornalVx = (2/Rx52**2)*(pointV[0] + Rx52 - Lensd5)
        nornalVy = (2/Rx52**2)*pointV[1]
        nornalVz = (2/Rx52**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    # スクリーンとの交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Screen(self, startV, directionV):
        T = (screenV[0]-startV[0])/directionV[0]
        return T

    # スネルの公式から屈折光の方向ベクトルを求める関数

    def decideRefractionVL(self, rayV, normalV, Nair, Nn):
        # 正規化
        rayV = rayV/np.linalg.norm(rayV)
        normalV = normalV/np.linalg.norm(normalV)
        # 係数A
        A = Nair/Nn
        # 入射角
        cos_t_in = -np.dot(rayV, normalV)
        # 量子化誤差対策
        if cos_t_in < -1.:
            cos_t_in = -1.
        elif cos_t_in > 1.:
            cos_t_in = 1.
        # スネルの法則
        sin_t_in = np.sqrt(1.0 - cos_t_in**2)
        sin_t_out = sin_t_in*A
        if sin_t_out > 1.0:
            # 全反射する場合
            return np.zeros(3)
        cos_t_out = np.sqrt(1 - sin_t_out**2)
        # 係数B
        B = -cos_t_out + A*cos_t_in
        # 出射光線の方向ベクトル
        outRayV = A*rayV + B*normalV
        # 正規化
        outRayV = outRayV/np.linalg.norm(outRayV)
        return outRayV

    def decideRefractionVR(self, rayV, normalV, Nair, Nn):
        # 正規化
        rayV = rayV/np.linalg.norm(rayV)
        normalV = normalV/np.linalg.norm(normalV)
        # 係数A
        A = Nair/Nn
        # 入射角
        cos_t_in = np.dot(rayV, normalV)
        # 量子化誤差対策
        if cos_t_in < -1.:
            cos_t_in = -1.
        elif cos_t_in > 1.:
            cos_t_in = 1.
        # スネルの法則
        sin_t_in = np.sqrt(1.0 - cos_t_in**2)
        sin_t_out = sin_t_in*A
        if sin_t_out > 1.0:
            # 全反射する場合
            return np.zeros(3)
        cos_t_out = np.sqrt(1 - sin_t_out**2)
        # 係数B
        B = -cos_t_out + A*cos_t_in
        # 出射光線の方向ベクトル
        outRayV = A*rayV + B*normalV
        # 正規化
        outRayV = outRayV/np.linalg.norm(outRayV)
        return outRayV

    # ２点の位置ベクトルから直線を引く関数
    def plotLineRed(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='r')

    def plotLinePurple(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='purple')

    def plotLineBlue(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='blue')

    def plotLineBlack(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='black')


# レンズと光線図
def MacroLens_Lens12345(RAD):
    # スクリーン描画
    Ys, Zs = np.meshgrid(
        np.arange(-3, 3.5, 0.5),
        np.arange(-3, 3.5, 0.5))
    Xs = 0*Ys + 0*Zs + screenV[0]
    ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    Xs = 0*Ys + 0*Zs + 10.695
    ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    Xs = 0*Ys + 0*Zs + 14.895
    ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    limitTheta = 2*np.pi  # theta生成数
    limitPhi = np.pi  # phi生成数
    theta = np.linspace(0, limitTheta, geneNum)
    phi = np.linspace(0, limitPhi, geneNum)

    # レンズ描画
    def plotLens(Rxi1, Rxi2, LensD, RLimit, LensType, lensiV):
        Ys = np.outer(np.sin(theta), np.sin(phi))
        Zs = np.outer(np.ones(np.size(theta)), np.cos(phi))

        Ysi1 = RLimit[0] * Ys
        Zsi1 = RLimit[0] * Zs
        Xsi1 = LensType[0] * (
            Rxi1**2-Ysi1**2-Zsi1**2)**(1/2) - LensType[0]*Rxi1 + lensiV[0]
        ax.plot_wireframe(Xsi1, Ysi1, Zsi1, linewidth=0.1)

        Ysi2 = RLimit[1] * Ys
        Zsi2 = RLimit[1] * Zs
        Xsi2 = LensType[1] * (
            Rxi2**2-Ysi2**2-Zsi2**2)**(1/2) + LensType[1]*(
            -Rxi2 + LensType[1]*LensD) + lensiV[0]
        ax.plot_wireframe(Xsi2, Ysi2, Zsi2, linewidth=0.1)

    for i in Params:
        plotLens(*i)

    VF = VectorFunctions()  # インスタンス化

    # 始点を生成する
    width = 3
    space = 1
    size = len(np.arange(-width+centerY, 1+width+centerY, space))**2
    pointsY, pointsZ = np.meshgrid(
        np.arange(-width+centerY, 1+width+centerY, space),
        np.arange(-width+centerZ, 1+width+centerZ, space))
    pointsX = np.array([centerX]*size) + lens1V[0]
    pointsY = pointsY.reshape(size)*rayDensity + lens1V[1]
    pointsZ = pointsZ.reshape(size)*rayDensity + lens1V[2]
    raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, size, 3)

    LastRedPoints = []
    LastBluePoints = []

    for i in raySPoint0:
        raySPoint0 = i
        # directionVector0 = np.array([1, 0, 0])  # 入射光の方向ベクトルを設定
        directionVector0 = rayStartV + raySPoint0 + \
            np.array([0*100, 0*100, 10000*np.sin(RAD)*100])
        T = VF.rayTraceDecideT_Lens1L(raySPoint0, directionVector0)  # 交点のための係数
        rayEPoint0 = raySPoint0 + T*directionVector0 + \
            np.array([0, 0, -2.5*np.sin(RAD)])  # 入射光の終点
        # 入射光描画
        VF.plotLinePurple(-rayStartV -
                          np.array([0*100, 0*100, 10000*np.sin(RAD)*100]), rayEPoint0)
        # VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

        # 赤色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(
            directionVector0, normalV_Lens1L, Nair, Nlens1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        VF.plotLineRed(refractSPoint0, refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(
            refractionV_Lens1L, normalV1, Nlens2, Nlens1)
        T = 0  # レンズの接着を考えた
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        VF.plotLineRed(raySPoint1, rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(
            refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(
            refractionV_Lens1R, normalV_Lens2L, Nlens1, Nlens2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        VF.plotLineRed(refractSPoint_Lens2L, refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(
            refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        VF.plotLineRed(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(
            refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(
            refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        VF.plotLineRed(refractSPoint_Lens3L, refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(
            refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3)
        T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        VF.plotLineRed(raySPoint_Lens3R, rayEPoint_Lens4L)
        # ax.quiver(raySPoint_Lens3R[0], raySPoint_Lens3R[1], raySPoint_Lens3R[2],
        #    refractionV_Lens3R[0], refractionV_Lens3R[1], refractionV_Lens3R[2])

        # 絞り
        rayEPoint_Lens4L_TEST = raySPoint_Lens3R + 0.85*refractionV_Lens3R
        TestR = rayEPoint_Lens4L_TEST[1]**2 + rayEPoint_Lens4L_TEST[2]**2
        if TestR >= 1.2:  # F値 5.6ほど
            refractionV_Lens3R = [0, 0, 0]

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(
            refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        VF.plotLineRed(refractSPoint_Lens4L, refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(
            refractionV_Lens4L, normalV_Lens4R, Nlens5, Nlens4)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
        VF.plotLineRed(raySPoint_Lens4R, rayEPoint_Lens5L)

        refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
        normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
        refractionV_Lens5L = VF.decideRefractionVL(
            refractionV_Lens4R, normalV_Lens5L, Nlens4, Nlens5)
        T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
        refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
        VF.plotLineRed(refractSPoint_Lens5L, refractEPoint_Lens5L)  # 屈折光の描画
        raySPoint_Lens5R = refractEPoint_Lens5L
        normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
        refractionV_Lens5R = VF.decideRefractionVR(
            refractionV_Lens5L, normalV_Lens5R, Nair, Nlens5)

        T = VF.rayTraceDecideT_Screen(raySPoint_Lens5R, refractionV_Lens5R)
        rayEPoint_Last = raySPoint_Lens5R + T*refractionV_Lens5R
        VF.plotLineRed(raySPoint_Lens5R, rayEPoint_Last)

        LastRedPoints.append(rayEPoint_Last)

        # 青色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(
            directionVector0, normalV_Lens1L, Nair, Nlens1*NBlueRay1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        VF.plotLineBlue(refractSPoint0, refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(
            refractionV_Lens1L, normalV1, Nlens2*NBlueRay2, Nlens1*NBlueRay1)
        T = 0  # レンズの接着を考えた
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        VF.plotLineBlue(raySPoint1, rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(
            refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(
            refractionV_Lens1R, normalV_Lens2L, Nlens1*NBlueRay1, Nlens2*NBlueRay2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        VF.plotLineBlue(refractSPoint_Lens2L, refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(
            refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2*NBlueRay2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        VF.plotLineBlue(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(
            refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(
            refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3*NBlueRay3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        VF.plotLineBlue(refractSPoint_Lens3L, refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(
            refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3*NBlueRay3)
        T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        VF.plotLineBlue(raySPoint_Lens3R, rayEPoint_Lens4L)

        # 絞り
        rayEPoint_Lens4L_TEST = raySPoint_Lens3R + 0.85*refractionV_Lens3R
        TestR = rayEPoint_Lens4L_TEST[1]**2 + rayEPoint_Lens4L_TEST[2]**2
        if TestR >= 1.2:  # F値 5.6ほど
            refractionV_Lens3R = [0, 0, 0]

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(
            refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4*NBlueRay4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        VF.plotLineBlue(refractSPoint_Lens4L, refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(
            refractionV_Lens4L, normalV_Lens4R, Nlens5*NBlueRay5, Nlens4*NBlueRay4)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
        VF.plotLineBlue(raySPoint_Lens4R, rayEPoint_Lens5L)

        refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
        normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
        refractionV_Lens5L = VF.decideRefractionVL(
            refractionV_Lens4R, normalV_Lens5L, Nlens4*NBlueRay4, Nlens5*NBlueRay5)
        T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
        refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
        VF.plotLineBlue(refractSPoint_Lens5L, refractEPoint_Lens5L)  # 屈折光の描画
        raySPoint_Lens5R = refractEPoint_Lens5L
        normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
        refractionV_Lens5R = VF.decideRefractionVR(
            refractionV_Lens5L, normalV_Lens5R, Nair, Nlens5*NBlueRay5)

        T = VF.rayTraceDecideT_Screen(raySPoint_Lens5R, refractionV_Lens5R)
        rayEPoint_Last = raySPoint_Lens5R + T*refractionV_Lens5R
        VF.plotLineBlue(raySPoint_Lens5R, rayEPoint_Last)

        LastBluePoints.append(rayEPoint_Last)

    if RAD == 0.00:
        ax.set_title('Ray Trace Layout 0.00 rad', pad=-7)
    elif RAD == 0.04:
        ax.set_title('0.04 rad', pad=-7)
    elif RAD == 0.07:
        ax.set_title('0.07 rad', pad=-7)
    ax.set_xlim(-LX-2, LX-2)
    ax.set_ylim(-LY, LY)
    ax.axes.yaxis.set_ticks([])
    ax.set_zlim(-LZ, LZ)
    ax.set_xlabel('x / cm')
    # ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=0, azim=-90)


# 円形スポットダイアグラム
def SpotDiagram_Circle(RAD):
    def T_FocusGraph(startV, directionV):
        T = -startV[2]/directionV[2]
        return T

    # スクリーン描画
    Ys, Zs = np.meshgrid(
        np.arange(-3, 3.5, 0.5),
        np.arange(-3, 3.5, 0.5))
    Xs = 0*Ys + 0*Zs + screenV[0]
    #ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    Xs = 0*Ys + 0*Zs + 10.695
    #ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    Xs = 0*Ys + 0*Zs + 14.895
    #ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    limitTheta = 2*np.pi  # theta生成数
    limitPhi = np.pi  # phi生成数
    theta = np.linspace(0, limitTheta, geneNum)
    phi = np.linspace(0, limitPhi, geneNum)

    # レンズ描画
    def plotLens(Rxi1, Rxi2, LensD, RLimit, LensType, lensiV):
        Ys = np.outer(np.sin(theta), np.sin(phi))
        Zs = np.outer(np.ones(np.size(theta)), np.cos(phi))

        Ysi1 = RLimit[0] * Ys
        Zsi1 = RLimit[0] * Zs
        Xsi1 = LensType[0] * (
            Rxi1**2-Ysi1**2-Zsi1**2)**(1/2) - LensType[0]*Rxi1 + lensiV[0]
        ax.plot_wireframe(Xsi1, Ysi1, Zsi1, linewidth=0.1)

        Ysi2 = RLimit[1] * Ys
        Zsi2 = RLimit[1] * Zs
        Xsi2 = LensType[1] * (
            Rxi2**2-Ysi2**2-Zsi2**2)**(1/2) + LensType[1]*(
            -Rxi2 + LensType[1]*LensD) + lensiV[0]
        ax.plot_wireframe(Xsi2, Ysi2, Zsi2, linewidth=0.1)

    # for i in Params:
    #    plotLens(*i)

    VF = VectorFunctions()  # インスタンス化

    # 始点を生成する
    width = 3

    pointsX = []
    pointsY = []
    pointsZ = []

    for i in range(50):
        r = width*i/50
        for j in range(12*4):
            theta = np.pi*j/(6*4)
            pointsY.append(r*np.cos(theta))
            pointsZ.append(r*np.sin(theta))
    pointsX = np.array([centerX]*len(pointsZ))

    raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, len(pointsZ), 3)

    RedFocusPoints = []
    BlueFocusPoints = []

    for i in raySPoint0:
        raySPoint0 = i
        # directionVector0 = np.array([1, 0, 0])  # 入射光の方向ベクトルを設定
        directionVector0 = rayStartV + raySPoint0 + \
            np.array([0*100, 0*100, 10000*np.sin(RAD)*100])
        T = VF.rayTraceDecideT_Lens1L(raySPoint0, directionVector0)  # 交点のための係数
        rayEPoint0 = raySPoint0 + T*directionVector0 + \
            np.array([0, 0, -2.5*np.sin(RAD)])  # 入射光の終点
        # VF.plotLinePurple(-rayStartV - np.array([0*100, 0*100, 10000*np.sin(RAD)*100]), rayEPoint0)  # 入射光描画
        # VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

        # 赤色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(
            directionVector0, normalV_Lens1L, Nair, Nlens1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        # VF.plotLineRed(refractSPoint0,refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(
            refractionV_Lens1L, normalV1, Nlens2, Nlens1)
        T = 0  # レンズの接着を考えた
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        # VF.plotLineRed(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(
            refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(
            refractionV_Lens1R, normalV_Lens2L, Nlens1, Nlens2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        # VF.plotLineRed(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(
            refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineRed(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(
            refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(
            refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        # VF.plotLineRed(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(
            refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3)
        T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        #VF.plotLineRed(raySPoint_Lens3R, rayEPoint_Lens4L)
        # ax.quiver(raySPoint_Lens3R[0], raySPoint_Lens3R[1], raySPoint_Lens3R[2],
        #    refractionV_Lens3R[0], refractionV_Lens3R[1], refractionV_Lens3R[2])

        # 絞り
        rayEPoint_Lens4L_TEST = raySPoint_Lens3R + 0.85*refractionV_Lens3R
        TestR = rayEPoint_Lens4L_TEST[1]**2 + rayEPoint_Lens4L_TEST[2]**2
        if TestR >= 1.2:  # F値 5.6ほど
            refractionV_Lens3R = [0, 0, 0]

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(
            refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        # VF.plotLineRed(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(
            refractionV_Lens4L, normalV_Lens4R, Nlens5, Nlens4)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
        #VF.plotLineRed(raySPoint_Lens4R, rayEPoint_Lens5L)

        refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
        normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
        refractionV_Lens5L = VF.decideRefractionVL(
            refractionV_Lens4R, normalV_Lens5L, Nlens4, Nlens5)
        T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
        refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
        # VF.plotLineRed(refractSPoint_Lens5L,refractEPoint_Lens5L)  # 屈折光の描画
        raySPoint_Lens5R = refractEPoint_Lens5L
        normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
        refractionV_Lens5R = VF.decideRefractionVR(
            refractionV_Lens5L, normalV_Lens5R, Nair, Nlens5)

        T = VF.rayTraceDecideT_Screen(raySPoint_Lens5R, refractionV_Lens5R)
        rayEPoint_Last = raySPoint_Lens5R + T*refractionV_Lens5R
        #VF.plotLineRed(raySPoint_Lens5R, rayEPoint_Last)

        RedFocusPoints.append(rayEPoint_Last)

        # 青色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(
            directionVector0, normalV_Lens1L, Nair, Nlens1*NBlueRay1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        # VF.plotLineBlue(refractSPoint0,refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(
            refractionV_Lens1L, normalV1, Nlens2*NBlueRay2, Nlens1*NBlueRay1)
        T = 0  # レンズの接着を考えた
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        # VF.plotLineBlue(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(
            refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(
            refractionV_Lens1R, normalV_Lens2L, Nlens1*NBlueRay1, Nlens2*NBlueRay2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        # VF.plotLineBlue(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(
            refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2*NBlueRay2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineBlue(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(
            refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(
            refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3*NBlueRay3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        # VF.plotLineBlue(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(
            refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3*NBlueRay3)
        T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        #VF.plotLineBlue(raySPoint_Lens3R, rayEPoint_Lens4L)

        # 絞り
        rayEPoint_Lens4L_TEST = raySPoint_Lens3R + 0.85*refractionV_Lens3R
        TestR = rayEPoint_Lens4L_TEST[1]**2 + rayEPoint_Lens4L_TEST[2]**2
        if TestR >= 1.2:  # F値 5.6ほど
            refractionV_Lens3R = [0, 0, 0]

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(
            refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4*NBlueRay4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        # VF.plotLineBlue(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(
            refractionV_Lens4L, normalV_Lens4R, Nlens5*NBlueRay5, Nlens4*NBlueRay4)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
        #VF.plotLineBlue(raySPoint_Lens4R, rayEPoint_Lens5L)

        refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
        normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
        refractionV_Lens5L = VF.decideRefractionVL(
            refractionV_Lens4R, normalV_Lens5L, Nlens4*NBlueRay4, Nlens5*NBlueRay5)
        T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
        refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
        # VF.plotLineBlue(refractSPoint_Lens5L,refractEPoint_Lens5L)  # 屈折光の描画
        raySPoint_Lens5R = refractEPoint_Lens5L
        normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
        refractionV_Lens5R = VF.decideRefractionVR(
            refractionV_Lens5L, normalV_Lens5R, Nair, Nlens5*NBlueRay5)

        T = VF.rayTraceDecideT_Screen(raySPoint_Lens5R, refractionV_Lens5R)
        rayEPoint_Last = raySPoint_Lens5R + T*refractionV_Lens5R
        #VF.plotLineBlue(raySPoint_Lens5R, rayEPoint_Last)

        BlueFocusPoints.append(rayEPoint_Last)

    for i in RedFocusPoints:
        ax.scatter(*i, color='r', s=1)

    for i in BlueFocusPoints:
        ax.scatter(*i, color='b', s=1)

    if RAD == 0.00:
        ax.set_title('Spot Diagram_Circle', pad=-7)
    ax.set_xlim(-LX/(-100*RAD+20) +
                RedFocusPoints[30][0], LX/(-100*RAD+20)+RedFocusPoints[30][0])
    ax.set_ylim(-LY/(-100*RAD+20), LY/(-100*RAD+20))
    ax.set_zlim(-LZ/(-100*RAD+20) +
                RedFocusPoints[30][2], LZ/(-100*RAD+20)+RedFocusPoints[30][2])
    # ax.set_xlabel('x')
    ax.axes.xaxis.set_ticks([])
    ax.set_ylabel('y / cm')
    ax.set_zlabel('z')
    ax.view_init(elev=0, azim=0)


# 四角スポットダイアグラム
def SpotDiagram_Square(RAD):
    def T_FocusGraph(startV, directionV):
        T = -startV[2]/directionV[2]
        return T

    # スクリーン描画
    Ys, Zs = np.meshgrid(
        np.arange(-3, 3.5, 0.5),
        np.arange(-3, 3.5, 0.5))
    Xs = 0*Ys + 0*Zs + screenV[0]
    #ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    Xs = 0*Ys + 0*Zs + 10.695
    #ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    Xs = 0*Ys + 0*Zs + 14.895
    #ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

    limitTheta = 2*np.pi  # theta生成数
    limitPhi = np.pi  # phi生成数
    theta = np.linspace(0, limitTheta, geneNum)
    phi = np.linspace(0, limitPhi, geneNum)

    # レンズ描画
    def plotLens(Rxi1, Rxi2, LensD, RLimit, LensType, lensiV):
        Ys = np.outer(np.sin(theta), np.sin(phi))
        Zs = np.outer(np.ones(np.size(theta)), np.cos(phi))

        Ysi1 = RLimit[0] * Ys
        Zsi1 = RLimit[0] * Zs
        Xsi1 = LensType[0] * (
            Rxi1**2-Ysi1**2-Zsi1**2)**(1/2) - LensType[0]*Rxi1 + lensiV[0]
        ax.plot_wireframe(Xsi1, Ysi1, Zsi1, linewidth=0.1)

        Ysi2 = RLimit[1] * Ys
        Zsi2 = RLimit[1] * Zs
        Xsi2 = LensType[1] * (
            Rxi2**2-Ysi2**2-Zsi2**2)**(1/2) + LensType[1]*(
            -Rxi2 + LensType[1]*LensD) + lensiV[0]
        ax.plot_wireframe(Xsi2, Ysi2, Zsi2, linewidth=0.1)

    # for i in Params:
    #    plotLens(*i)

    VF = VectorFunctions()  # インスタンス化

    # 始点を生成する
    width = 3
    space = 0.5
    size = len(np.arange(-width+centerY, 1+width+centerY, space))**2
    pointsY, pointsZ = np.meshgrid(
        np.arange(-width+centerY, 1+width+centerY, space),
        np.arange(-width+centerZ, 1+width+centerZ, space))
    pointsX = np.array([centerX]*size) + lens1V[0]
    pointsY = pointsY.reshape(size)*rayDensity + lens1V[1]
    pointsZ = pointsZ.reshape(size)*rayDensity + lens1V[2]
    raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, size, 3)

    RedFocusPoints = []
    BlueFocusPoints = []

    for i in raySPoint0:
        raySPoint0 = i
        # directionVector0 = np.array([1, 0, 0])  # 入射光の方向ベクトルを設定
        directionVector0 = rayStartV + raySPoint0 + \
            np.array([0*100, 0*100, 10000*np.sin(RAD)*100])
        T = VF.rayTraceDecideT_Lens1L(raySPoint0, directionVector0)  # 交点のための係数
        rayEPoint0 = raySPoint0 + T*directionVector0 + \
            np.array([0, 0, -2.5*np.sin(RAD)])  # 入射光の終点
        # VF.plotLinePurple(-rayStartV - np.array([0*100, 0*100, 10000*np.sin(RAD)*100]), rayEPoint0)  # 入射光描画
        # VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

        # 赤色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(
            directionVector0, normalV_Lens1L, Nair, Nlens1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        # VF.plotLineRed(refractSPoint0,refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(
            refractionV_Lens1L, normalV1, Nlens2, Nlens1)
        T = 0  # レンズの接着を考えた
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        # VF.plotLineRed(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(
            refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(
            refractionV_Lens1R, normalV_Lens2L, Nlens1, Nlens2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        # VF.plotLineRed(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(
            refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineRed(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(
            refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(
            refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        # VF.plotLineRed(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(
            refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3)
        T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        #VF.plotLineRed(raySPoint_Lens3R, rayEPoint_Lens4L)
        # ax.quiver(raySPoint_Lens3R[0], raySPoint_Lens3R[1], raySPoint_Lens3R[2],
        #    refractionV_Lens3R[0], refractionV_Lens3R[1], refractionV_Lens3R[2])

        # 絞り
        rayEPoint_Lens4L_TEST = raySPoint_Lens3R + 0.85*refractionV_Lens3R
        TestR = rayEPoint_Lens4L_TEST[1]**2 + rayEPoint_Lens4L_TEST[2]**2
        if TestR >= 1.2:  # F値 5.6ほど
            refractionV_Lens3R = [0, 0, 0]

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(
            refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        # VF.plotLineRed(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(
            refractionV_Lens4L, normalV_Lens4R, Nlens5, Nlens4)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
        #VF.plotLineRed(raySPoint_Lens4R, rayEPoint_Lens5L)

        refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
        normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
        refractionV_Lens5L = VF.decideRefractionVL(
            refractionV_Lens4R, normalV_Lens5L, Nlens4, Nlens5)
        T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
        refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
        # VF.plotLineRed(refractSPoint_Lens5L,refractEPoint_Lens5L)  # 屈折光の描画
        raySPoint_Lens5R = refractEPoint_Lens5L
        normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
        refractionV_Lens5R = VF.decideRefractionVR(
            refractionV_Lens5L, normalV_Lens5R, Nair, Nlens5)

        T = VF.rayTraceDecideT_Screen(raySPoint_Lens5R, refractionV_Lens5R)
        rayEPoint_Last = raySPoint_Lens5R + T*refractionV_Lens5R
        #VF.plotLineRed(raySPoint_Lens5R, rayEPoint_Last)

        RedFocusPoints.append(rayEPoint_Last)

        # 青色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(
            directionVector0, normalV_Lens1L, Nair, Nlens1*NBlueRay1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        # VF.plotLineBlue(refractSPoint0,refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(
            refractionV_Lens1L, normalV1, Nlens2*NBlueRay2, Nlens1*NBlueRay1)
        T = 0  # レンズの接着を考えた
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        # VF.plotLineBlue(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(
            refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(
            refractionV_Lens1R, normalV_Lens2L, Nlens1*NBlueRay1, Nlens2*NBlueRay2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        # VF.plotLineBlue(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(
            refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2*NBlueRay2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineBlue(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(
            refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(
            refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3*NBlueRay3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        # VF.plotLineBlue(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(
            refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3*NBlueRay3)
        T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        #VF.plotLineBlue(raySPoint_Lens3R, rayEPoint_Lens4L)

        # 絞り
        rayEPoint_Lens4L_TEST = raySPoint_Lens3R + 0.85*refractionV_Lens3R
        TestR = rayEPoint_Lens4L_TEST[1]**2 + rayEPoint_Lens4L_TEST[2]**2
        if TestR >= 1.2:  # F値 5.6ほど
            refractionV_Lens3R = [0, 0, 0]

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(
            refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4*NBlueRay4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        # VF.plotLineBlue(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(
            refractionV_Lens4L, normalV_Lens4R, Nlens5*NBlueRay5, Nlens4*NBlueRay4)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
        #VF.plotLineBlue(raySPoint_Lens4R, rayEPoint_Lens5L)

        refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
        normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
        refractionV_Lens5L = VF.decideRefractionVL(
            refractionV_Lens4R, normalV_Lens5L, Nlens4*NBlueRay4, Nlens5*NBlueRay5)
        T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
        refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
        # VF.plotLineBlue(refractSPoint_Lens5L,refractEPoint_Lens5L)  # 屈折光の描画
        raySPoint_Lens5R = refractEPoint_Lens5L
        normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
        refractionV_Lens5R = VF.decideRefractionVR(
            refractionV_Lens5L, normalV_Lens5R, Nair, Nlens5*NBlueRay5)

        T = VF.rayTraceDecideT_Screen(raySPoint_Lens5R, refractionV_Lens5R)
        rayEPoint_Last = raySPoint_Lens5R + T*refractionV_Lens5R
        #VF.plotLineBlue(raySPoint_Lens5R, rayEPoint_Last)

        BlueFocusPoints.append(rayEPoint_Last)

    for i in RedFocusPoints:
        ax.scatter(*i, color='r', s=1)

    for i in BlueFocusPoints:
        ax.scatter(*i, color='b', s=1)

    if RAD == 0.00:
        ax.set_title('Spot Diagram_Square', pad=-7)
    ax.set_xlim(-LX/(-100*RAD+20) +
                RedFocusPoints[25][0], LX/(-100*RAD+20)+RedFocusPoints[25][0])
    ax.set_ylim(-LY/(-100*RAD+20), LY/(-100*RAD+20))
    ax.set_zlim(-LZ/(-100*RAD+20) +
                RedFocusPoints[25][2], LZ/(-100*RAD+20)+RedFocusPoints[25][2])
    # ax.set_xlabel('x')
    ax.axes.xaxis.set_ticks([])
    ax.set_ylabel('y / cm')
    ax.set_zlabel('z')
    ax.view_init(elev=0, azim=0)


if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()
    fig = plt.figure(figsize=(9, 9))
    plt.rcParams["font.size"] = 8

    ax = fig.add_subplot(3, 3, 1, projection='3d')
    MacroLens_Lens12345(0.00)
    ax = fig.add_subplot(3, 3, 2, projection='3d')
    SpotDiagram_Circle(0.00)
    ax = fig.add_subplot(3, 3, 3, projection='3d')
    SpotDiagram_Square(0.00)

    ax = fig.add_subplot(3, 3, 4, projection='3d')
    MacroLens_Lens12345(0.04)
    ax = fig.add_subplot(3, 3, 5, projection='3d')
    SpotDiagram_Circle(0.04)
    ax = fig.add_subplot(3, 3, 6, projection='3d')
    SpotDiagram_Square(0.04)

    ax = fig.add_subplot(3, 3, 7, projection='3d')
    MacroLens_Lens12345(0.07)
    ax = fig.add_subplot(3, 3, 8, projection='3d')
    SpotDiagram_Circle(0.07)
    ax = fig.add_subplot(3, 3, 9, projection='3d')
    SpotDiagram_Square(0.07)

    print('\ntime =', time.time()-start)
    print('\n----------------END----------------\n')
    plt.show()
