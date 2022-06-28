# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time


LX = 4
LY = 4
LZ = 4
geneNum = 500
Nair = 1  # 空気の屈折率

rayStartV = np.array([10000*100, 0*100, 0.0*100])  # m から cm へ変換
centerX = 0  # 入射光表示の中心座標
centerY = 0  # 入射光表示の中心座標
centerZ = 0  # 入射光表示の中心座標
rayDensity = 0.25  # 入射光の密度

# ------------------------入力ここから---------------------------
screenV = np.array([14.1993, 0, 0])  # スクリーンの位置ベクトル
UnitX = -0

"""
lens1V = np.array([0+UnitX, 0, 0])  # レンズ１の位置ベクトル
lens2V = np.array([0+4+UnitX, 0, 0])  # レンズ2の位置ベクトル
lens3V = np.array([0+6+UnitX, 0, 0])  # レンズ3の位置ベクトル

Lens1Param = [10, 1000000000000, [-1, 1], 1, lens1V, [4.359, 4.359]]
Lens2Param = [10, 1000000000000, [-1, 1], 1, lens2V, [4.359, 4.359]]
Lens3Param = [1000000000000, 10, [-1, 1], 1, lens3V, [4.359, 4.359]]

LensParams = np.array([Lens1Param, Lens2Param, Lens3Param])

paramList = [
    [1.61727, 1.62041, 1.62756],  # N-SK16
    [1.61727, 1.62041, 1.62756],  # N-SK16
    [1.61727, 1.62041, 1.62756]   # N-SK16
]

"""
lens1V = np.array([0+UnitX, 0, 0])  # レンズ１の位置ベクトル
lens2V = np.array([0.756+UnitX, 0, 0])  # レンズ２の位置ベクトル
lens3V = np.array([1.886+UnitX, 0, 0])  # レンズ３の位置ベクトル
lens4V = np.array([3.676+UnitX, 0, 0])  # レンズ４の位置ベクトル
lens5V = np.array([3.876+UnitX, 0, 0])  # レンズ5の位置ベクトル

# Radius_L, Radius_R, [lenstype_L, lenstype_R], lens_d, lens_point, [aperture_L, aperture_R]
Lens1Param = np.array([4.78, 9.3, [-1, 1], 0.756, lens1V, [1.791, 1.791]])
Lens2Param = np.array([9.3, 26.25, [1, -1], 0.27, lens2V, [1.788, 1.788]])
Lens3Param = np.array([11.27, 4.3, [1, -1], 0.514, lens3V, [1.26, 1.26]])
Lens4Param = np.array([1000000, 6.58, [1, -1], 0.2, lens4V, [1.395, 1.395]])
Lens5Param = np.array([6.58, 9.305, [-1, 1], 0.58, lens5V, [1.395, 1.395]])

LensParams = np.array([Lens1Param, Lens2Param, Lens3Param,
                      Lens4Param, Lens5Param])  # これを関数に渡す

# [nC, nd, nF]  GlassName
paramList = np.array([
    [1.89526, 1.90366, 1.92411],  # N-LASF46A_SCHOTT  RefCODEV探索結果
    [1.66664, 1.67271, 1.68750],  # SF5_SCHOTT
    [1.91039, 1.92286, 1.95459],  # N-SF66_SCHOTT
    [1.51982, 1.52249, 1.52860],  # N-K5_SCHOTT
    [1.79901, 1.80420, 1.81630]   # N-LASF44_SCHOTT
])


# ---------------------------入力ここまで------------------------


# 法線ベクトル関数とレイトレース関数に使うshiftVの計算
def make_all_shift(LensParams):
    all_shift = []
    for i in LensParams:
        # -lensType_L*radius_L
        # -lensType_R*radius_R + lens_d
        shift_lensiL = np.array(i[4] + [-i[2][0]*i[0], 0, 0])
        shift_lensiR = np.array(i[4] + [-i[2][1]*i[1]+i[3], 0, 0])
        all_shift.append(shift_lensiL)
        all_shift.append(shift_lensiR)
    return all_shift


all_shift = make_all_shift(LensParams)

# 曲率半径リスト作成


def make_radius(LensParams):
    radius = []
    for i in LensParams:
        radius.append(i[0])
        radius.append(i[1])
    return radius


all_radius = make_radius(LensParams)

# レンズタイプリスト作成


def make_lenstype(LensParams):
    lenstype = []
    for i in LensParams:
        lenstype.append(i[2][0])
        lenstype.append(i[2][1])
    return lenstype


all_lenstype = make_lenstype(LensParams)

# 屈折率のリストを作成


def make_nC(paramList):
    nC = []
    for i in paramList:
        nC.append(i[0])
    return nC


all_nC = make_nC(paramList)
#print(all_nC, "nC")


def make_nd(paramList):
    nd = []
    for i in paramList:
        nd.append(i[1])
    return nd


all_nd = make_nd(paramList)
#print(all_nd, "nd")


def make_nF(paramList):
    nF = []
    for i in paramList:
        nF.append(i[2])
    return nF


all_nF = make_nF(paramList)
#print(all_nF, "nF")


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

    # 法線ベクトル関数（一般化）
    def decideNormalV(self, pointV, shiftV):
        return pointV-shiftV

    # レイトレース関数（一般化）
    def rayTrace_T(self, startV, directionV, shiftV, radius, lenstype):
        startV = startV-shiftV
        A = np.dot(directionV, directionV)
        B = np.dot(startV, directionV)
        C = np.dot(startV, startV) - radius**2
        T = (-B + lenstype*np.sqrt(B**2 - A*C))/A
        return [T]*3

    # スクリーンとの交点を持つときの係数Ｔを求める関数

    def rayTraceDecideT_Screen(self, startV, directionV):
        T = (screenV[0]-startV[0])/directionV[0]
        return [T]*3

    # スネルの法則から方向ベクトルを求める関数

    def decideRefractionV(self, rayV, normalV, Nin, Nout):
        if normalV[0] <= 0:
            # スネルの法則から屈折光の方向ベクトルを求める関数(左に凸の場合)
            # 正規化
            rayV = rayV/np.linalg.norm(rayV)
            normalV = normalV/np.linalg.norm(normalV)
            # 係数A
            A = Nin/Nout
            # 入射角
            cos_t_in = abs(np.dot(rayV, normalV))
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
            B = A*cos_t_in - cos_t_out
            # 出射光線の方向ベクトル
            outRayV = A*rayV + B*normalV
            # 正規化
            outRayV = outRayV/np.linalg.norm(outRayV)
        else:
            # スネルの法則から屈折光の方向ベクトルを求める関数(右に凸の場合)
            # 正規化
            rayV = rayV/np.linalg.norm(rayV)
            normalV = normalV/np.linalg.norm(normalV)
            # 係数A
            A = Nin/Nout
            # 入射角
            cos_t_in = abs(np.dot(rayV, normalV))
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
            B = -A*cos_t_in + cos_t_out
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

    def plotLineOrange(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='orange')

    def plotLineBlack(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='black')


# レンズ内光線描画
def DrawLens():

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
    def plotLens(Rxi1, Rxi2, LensType, LensD, lensiV, RLimit):
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

    for i in LensParams:
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
    raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, size, 3)  # 入射光の始点
    raySize = len(raySPoint0)

    # --------------------------------------------------------------------------------------

    directionVector0 = np.array([[1, 0, 0]]*raySize)  # 入射光の方向ベクトル
    #directionVector0 = rayStartV + raySPoint0
    T = np.array(list(map(VF.rayTrace_T, raySPoint0, directionVector0, [all_shift[0]]*raySize, [
                 all_radius[0]]*raySize, [all_lenstype[0]]*raySize))).reshape(raySize, 3)  # 交点のための係数
    rayEPoint0 = raySPoint0 + T*directionVector0  # 入射光の終点
    # 入射光描画
    for (i, j) in zip(raySPoint0-rayStartV, rayEPoint0):
        VF.plotLinePurple(i, j)
    # VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

    # レンズレイヤー
    def lenslayer(all_nX, line_color):
        for layer_num in range(0, len(all_shift), 2):
            # print(layer_num)
            # 一般化レイヤ, レンズ１つ分
            if layer_num == 0:
                raySPoint_lensL = rayEPoint0
                last_directionV = directionVector0
            else:
                raySPoint_lensL = last_Point
                last_directionV = last_directionV
            normalV_lensL = VF.decideNormalV(
                raySPoint_lensL, all_shift[layer_num])  # OK
            refractionV_lensL = np.array(list(map(VF.decideRefractionV, last_directionV, normalV_lensL, [
                                         Nair]*raySize, [all_nX[int(layer_num/2)]]*raySize)))  # OK
            T = np.array(list(map(VF.rayTrace_T, raySPoint_lensL, refractionV_lensL, [all_shift[layer_num+1]]*raySize, [
                         all_radius[layer_num+1]]*raySize, [all_lenstype[layer_num+1]]*raySize))).reshape(raySize, 3)  # OK
            refractEPoint_lensL = raySPoint_lensL + T*refractionV_lensL
            if line_color == 'red':
                for (i, j) in zip(raySPoint_lensL, refractEPoint_lensL):
                    VF.plotLineRed(i, j)
            elif line_color == 'blue':
                for (i, j) in zip(raySPoint_lensL, refractEPoint_lensL):
                    VF.plotLineBlue(i, j)
            elif line_color == 'orange':
                for (i, j) in zip(raySPoint_lensL, refractEPoint_lensL):
                    VF.plotLineOrange(i, j)

            raySPoint_lensR = refractEPoint_lensL
            normalV_lensR = VF.decideNormalV(
                raySPoint_lensR, all_shift[layer_num+1])  # OK
            refractionV_lensR = np.array(list(map(VF.decideRefractionV, refractionV_lensL, normalV_lensR, [
                                         all_nX[int(layer_num/2)]]*raySize, [Nair]*raySize)))  # OK
            if layer_num+2 == len(all_shift):
                T = np.array(list(map(VF.rayTraceDecideT_Screen,
                             raySPoint_lensR, refractionV_lensR))).reshape(raySize, 3)
                refractEPoint_lensR = raySPoint_lensR + T*refractionV_lensR
            else:
                T = np.array(list(map(VF.rayTrace_T, raySPoint_lensR, refractionV_lensR, [all_shift[layer_num+2]]*raySize, [
                             all_radius[layer_num+2]]*raySize, [all_lenstype[layer_num+2]]*raySize))).reshape(raySize, 3)  # OK
                refractEPoint_lensR = raySPoint_lensR + T*refractionV_lensR
            if line_color == 'red':
                for (i, j) in zip(raySPoint_lensR, refractEPoint_lensR):
                    VF.plotLineRed(i, j)
            elif line_color == 'blue':
                for (i, j) in zip(raySPoint_lensR, refractEPoint_lensR):
                    VF.plotLineBlue(i, j)
            elif line_color == 'orange':
                for (i, j) in zip(raySPoint_lensR, refractEPoint_lensR):
                    VF.plotLineOrange(i, j)

            last_Point = refractEPoint_lensR
            last_directionV = refractionV_lensR

    # C線
    #lenslayer(all_nC, "red")

    # d線
    lenslayer(all_nd, "orange")

    # F線
    #lenslayer(all_nF, "blue")

    # -----------------------------------------------------------------------------------
    ax.set_xlim(-LX+1.5, LX+1.5)
    ax.set_ylim(-LY, LY)
    ax.set_zlim(-LZ, LZ)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=0, azim=-90)

# スポットダイアグラム_同心円


def SpotDiagram_Circle():
    def T_FocalLength(startV, directionV):
        T = -startV[2]/directionV[2]
        return [T]*3

    VF = VectorFunctions()  # インスタンス化

    # 始点を生成する
    width = 3

    pointsX = []
    pointsY = []
    pointsZ = []

    for i in range(5):
        r = width*i/5
        for j in range(12*1):
            theta = np.pi*j/(6*1)
            pointsY.append(r*np.cos(theta))
            pointsZ.append(r*np.sin(theta))
    pointsX = np.array([centerX]*len(pointsZ))

    raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, len(pointsZ), 3)

    raySize = len(raySPoint0)
    print("raySPoint_Circle", raySize)
    directionVector0 = np.array([[1, 0, 0]]*raySize)
    #directionVector0 = rayStartV + raySPoint0
    T = np.array(list(map(VF.rayTrace_T, raySPoint0, directionVector0, [
                 all_shift[0]]*raySize, [all_radius[0]]*raySize, [all_lenstype[0]]*raySize))).reshape(raySize, 3)
    rayEPoint0 = raySPoint0 + T*directionVector0

    # レンズレイヤー 焦点と光軸からの距離の関係

    def lenslayer_focusFunc(all_nX, line_color):
        for layer_num in range(0, len(all_shift), 2):
            # 一般化レイヤ, レンズ１つ分
            if layer_num == 0:
                raySPoint_lensL = rayEPoint0
                last_directionV = directionVector0
            else:
                raySPoint_lensL = last_Point
                last_directionV = last_directionV
            normalV_lensL = VF.decideNormalV(
                raySPoint_lensL, all_shift[layer_num])  # OK
            refractionV_lensL = np.array(list(map(VF.decideRefractionV, last_directionV, normalV_lensL, [
                                         Nair]*raySize, [all_nX[int(layer_num/2)]]*raySize)))  # OK
            T = np.array(list(map(VF.rayTrace_T, raySPoint_lensL, refractionV_lensL, [all_shift[layer_num+1]]*raySize, [
                         all_radius[layer_num+1]]*raySize, [all_lenstype[layer_num+1]]*raySize))).reshape(raySize, 3)  # OK
            refractEPoint_lensL = raySPoint_lensL + T*refractionV_lensL

            raySPoint_lensR = refractEPoint_lensL
            normalV_lensR = VF.decideNormalV(
                raySPoint_lensR, all_shift[layer_num+1])  # OK
            refractionV_lensR = np.array(list(map(VF.decideRefractionV, refractionV_lensL, normalV_lensR, [
                                         all_nX[int(layer_num/2)]]*raySize, [Nair]*raySize)))  # OK
            if layer_num+2 == len(all_shift):
                #print("last layer")
                T = np.array(list(map(VF.rayTraceDecideT_Screen,
                             raySPoint_lensR, refractionV_lensR))).reshape(raySize, 3)

                focusPoint = raySPoint_lensR + T*refractionV_lensR
                FocusPoints = focusPoint.reshape(raySize*3)[::3]
                #print('Focus Point =', round(FocusPoints[0], 4), 'cm')

                if line_color == 'red':
                    for i in focusPoint:
                        ax.scatter(*i, color='r', s=0.5)
                    ax.set_xlim(-0.8+FocusPoints[0], 0.8+FocusPoints[0])
                    ax.set_ylim(-0.3, 0.3)
                    ax.set_zlim(-0.3, 0.3)
                    ax.set_xlabel('X / cm')
                    ax.set_ylabel('Y / cm')
                    ax.set_zlabel('Z / cm')
                    ax.view_init(elev=0, azim=-0)
                elif line_color == 'blue':
                    for i in focusPoint:
                        ax.scatter(*i, color='b', s=0.5)
                    ax.set_xlim(-0.8+FocusPoints[0], 0.8+FocusPoints[0])
                    ax.set_ylim(-0.3, 0.3)
                    ax.set_zlim(-0.3, 0.3)
                    ax.set_xlabel('X / cm')
                    ax.set_ylabel('Y / cm')
                    ax.set_zlabel('Z / cm')
                    ax.view_init(elev=0, azim=-0)
                elif line_color == 'orange':
                    for i in focusPoint:
                        ax.scatter(*i, color='orange', s=0.5)
                    ax.set_xlim(-0.8+FocusPoints[0], 0.8+FocusPoints[0])
                    ax.set_ylim(-0.3, 0.3)
                    ax.set_zlim(-0.3, 0.3)
                    ax.set_xlabel('X / cm')
                    ax.set_ylabel('Y / cm')
                    ax.set_zlabel('Z / cm')
                    ax.view_init(elev=0, azim=-0)
            else:
                T = np.array(list(map(VF.rayTrace_T, raySPoint_lensR, refractionV_lensR, [all_shift[layer_num+2]]*raySize, [
                             all_radius[layer_num+2]]*raySize, [all_lenstype[layer_num+2]]*raySize))).reshape(raySize, 3)  # OK
            refractEPoint_lensR = raySPoint_lensR + T*refractionV_lensR

            last_Point = refractEPoint_lensR
            last_directionV = refractionV_lensR

    # C線
    lenslayer_focusFunc(all_nC, 'red')

    # d線
    lenslayer_focusFunc(all_nd, 'orange')

    # F線
    lenslayer_focusFunc(all_nF, 'blue')

# スポットダイアグラム_正方形


def SpotDiagram_Square():
    def T_FocalLength(startV, directionV):
        T = -startV[2]/directionV[2]
        return [T]*3

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

    raySize = len(raySPoint0)
    print("raySPoint_Square", raySize)
    directionVector0 = np.array([[1, 0, 0]]*raySize)
    #directionVector0 = rayStartV + raySPoint0
    T = np.array(list(map(VF.rayTrace_T, raySPoint0, directionVector0, [
                 all_shift[0]]*raySize, [all_radius[0]]*raySize, [all_lenstype[0]]*raySize))).reshape(raySize, 3)
    rayEPoint0 = raySPoint0 + T*directionVector0

    # レンズレイヤー 焦点と光軸からの距離の関係

    def lenslayer_focusFunc(all_nX, line_color):
        for layer_num in range(0, len(all_shift), 2):
            # 一般化レイヤ, レンズ１つ分
            if layer_num == 0:
                raySPoint_lensL = rayEPoint0
                last_directionV = directionVector0
            else:
                raySPoint_lensL = last_Point
                last_directionV = last_directionV
            normalV_lensL = VF.decideNormalV(
                raySPoint_lensL, all_shift[layer_num])  # OK
            refractionV_lensL = np.array(list(map(VF.decideRefractionV, last_directionV, normalV_lensL, [
                                         Nair]*raySize, [all_nX[int(layer_num/2)]]*raySize)))  # OK
            T = np.array(list(map(VF.rayTrace_T, raySPoint_lensL, refractionV_lensL, [all_shift[layer_num+1]]*raySize, [
                         all_radius[layer_num+1]]*raySize, [all_lenstype[layer_num+1]]*raySize))).reshape(raySize, 3)  # OK
            refractEPoint_lensL = raySPoint_lensL + T*refractionV_lensL

            raySPoint_lensR = refractEPoint_lensL
            normalV_lensR = VF.decideNormalV(
                raySPoint_lensR, all_shift[layer_num+1])  # OK
            refractionV_lensR = np.array(list(map(VF.decideRefractionV, refractionV_lensL, normalV_lensR, [
                                         all_nX[int(layer_num/2)]]*raySize, [Nair]*raySize)))  # OK
            if layer_num+2 == len(all_shift):
                #print("last layer")
                T = np.array(list(map(VF.rayTraceDecideT_Screen,
                             raySPoint_lensR, refractionV_lensR))).reshape(raySize, 3)

                focusPoint = raySPoint_lensR + T*refractionV_lensR
                FocusPoints = focusPoint.reshape(raySize*3)[::3]
                #print('Focus Point =', round(FocusPoints[0], 4), 'cm')

                if line_color == 'red':
                    for i in focusPoint:
                        ax.scatter(*i, color='r', s=0.5)
                    ax.set_xlim(-0.8+FocusPoints[0], 0.8+FocusPoints[0])
                    ax.set_ylim(-0.01, 0.01)
                    ax.set_zlim(-0.01, 0.01)
                    ax.set_xlabel('X / cm')
                    ax.set_ylabel('Y / cm')
                    ax.set_zlabel('Z / cm')
                    ax.view_init(elev=0, azim=-0)
                elif line_color == 'blue':
                    for i in focusPoint:
                        ax.scatter(*i, color='b', s=0.5)
                    ax.set_xlim(-0.8+FocusPoints[0], 0.8+FocusPoints[0])
                    ax.set_ylim(-0.01, 0.01)
                    ax.set_zlim(-0.01, 0.01)
                    ax.set_xlabel('X / cm')
                    ax.set_ylabel('Y / cm')
                    ax.set_zlabel('Z / cm')
                    ax.view_init(elev=0, azim=-0)
                elif line_color == 'orange':
                    for i in focusPoint:
                        ax.scatter(*i, color='orange', s=0.5)
                    ax.set_xlim(-0.8+FocusPoints[0], 0.8+FocusPoints[0])
                    ax.set_ylim(-0.01, 0.01)
                    ax.set_zlim(-0.01, 0.01)
                    ax.set_xlabel('X / cm')
                    ax.set_ylabel('Y / cm')
                    ax.set_zlabel('Z / cm')
                    ax.view_init(elev=0, azim=-0)
            else:
                T = np.array(list(map(VF.rayTrace_T, raySPoint_lensR, refractionV_lensR, [all_shift[layer_num+2]]*raySize, [
                             all_radius[layer_num+2]]*raySize, [all_lenstype[layer_num+2]]*raySize))).reshape(raySize, 3)  # OK
            refractEPoint_lensR = raySPoint_lensR + T*refractionV_lensR

            last_Point = refractEPoint_lensR
            last_directionV = refractionV_lensR

    # C線
    lenslayer_focusFunc(all_nC, 'red')

    # d線
    lenslayer_focusFunc(all_nd, 'orange')

    # F線
    lenslayer_focusFunc(all_nF, 'blue')


if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()
    fig = plt.figure(figsize=(18, 6))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    DrawLens()

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    SpotDiagram_Circle()

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    SpotDiagram_Square()

    print('\ntime =', round(time.time()-start, 5), 'sec')
    print('\n----------------END----------------\n')
    plt.show()
