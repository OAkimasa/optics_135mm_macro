# Out_GlassListの比較をして、一致した項目を出力する
from Out_GlassList_after_inf12345 import GlassList_inf12345
from Out_GlassList_after_inf12 import GlassList_inf12
from Out_GlassList_after_inf12re import GlassList_inf12re
from Out_GlassList_after_inf45 import GlassList_inf45
from Out_GlassList_after_inf45re import GlassList_inf45re
from GlassData import GlassData


GlassList_inf12345 = GlassList_inf12345()
GlassList_inf12 = GlassList_inf12()
GlassList_inf12re = GlassList_inf12re()
GlassList_inf45 = GlassList_inf45()
GlassList_inf45re = GlassList_inf45re()

# --------------------------------------------------------------------------

print('\n------Compare-START-------\n')


# レンズ12345とレンズ12の一覧を比較
new_GlassList_12345and12 = []
for i in GlassList_inf12:
    for j in GlassList_inf12345:
        if i[0]==j[0] and i[1]==j[1]:
            new_GlassList_12345and12.append(i)

print('Compare Lens12345 & Lens12')
print('Lens12_Count =', len(GlassList_inf12))
print('Lens12345_Count =', len(GlassList_inf12345))
print('Match =', len(new_GlassList_12345and12))
#print(new_GlassList_12345and12)


# レンズ12345とレンズ12reの一覧を比較
new_GlassList_12345and12re = []
for i in GlassList_inf12re:
    for j in GlassList_inf12345:
        if i[0]==j[0] and i[1]==j[1]:
            new_GlassList_12345and12re.append(i)

print('\nCompare Lens12345 & Lens12re')
print('Lens12re_Count =', len(GlassList_inf12re))
print('Lens12345_Count =', len(GlassList_inf12345))
print('Match =', len(new_GlassList_12345and12re))
#print(new_GlassList_12345and12re)



# レンズ12とレンズ12reの一覧を比較
new_GlassList_12and12re = []
for i in GlassList_inf12:
    for j in GlassList_inf12re:
        if i[0]==j[0] and i[1]==j[1]:
            new_GlassList_12and12re.append(i)

#print(new_GlassList)
print('\nCompare Lens12 & Lens12re')
print('Lens12_Count =', len(GlassList_inf12))
print('Lens12re_Count =', len(GlassList_inf12re))
print('Match =', len(new_GlassList_12and12re))

# --------------------------------------------------------------------------

# レンズ12345とレンズ45の一覧を比較
new_GlassList_12345and45 = []
for i in GlassList_inf12345:
    for j in GlassList_inf45:
        if i[3]==j[0] and i[4]==j[1]:
            new_GlassList_12345and45.append(i)

#print(new_GlassList)
print('\nCompare Lens12345 & Lens45')
print('Lens45_Count =', len(GlassList_inf45))
print('Lens12345_Count =', len(GlassList_inf12345))
print('Match =', len(new_GlassList_12345and45))



# レンズ12345とレンズ45reの一覧を比較
new_GlassList_12345and45re = []
for i in GlassList_inf12345:
    for j in GlassList_inf45re:
        if i[3]==j[0] and i[4]==j[1]:
            new_GlassList_12345and45re.append(i)

#print(new_GlassList)
print('\nCompare Lens12345 & Lens45re')
print('Lens45re_Count =', len(GlassList_inf45re))
print('Lens12345_Count =', len(GlassList_inf12345))
print('Match =', len(new_GlassList_12345and45re))


# レンズ45とレンズ45reの一覧を比較
new_GlassList_45and45 = []
for i in GlassList_inf45:
    for j in GlassList_inf45re:
        if i[0]==j[0] and i[1]==j[1]:
            new_GlassList_45and45.append(i)

#print(new_GlassList)
print('\nCompare Lens45 & Lens45re')
print('Lens45_Count =', len(GlassList_inf45))
print('Lens45re_Count =', len(GlassList_inf45re))
print('Match =', len(new_GlassList_45and45))


# --------------------------------------------------------------------------


# レンズ12の候補リストを結合
new_GlassList_Lens12 = new_GlassList_12345and12 + new_GlassList_12345and12re
#print(new_GlassList_Lens12)
#print(len(new_GlassList_Lens12))


# レンズ3の候補リスト
new_GlassList_Lens3 = GlassData()
#print(new_GlassList_Lens3)
#print(len(new_GlassList_Lens3))


# レンズ45の候補リストを結合
new_GlassList_Lens45 = GlassList_inf45 + new_GlassList_12345and45re
#print(new_GlassList_Lens45)
#print(len(new_GlassList_Lens45))


# 出力用リスト作成
after_Compare_GlassList = []


after_Compare_GlassList123 = []

for i in new_GlassList_Lens12:
    for j in new_GlassList_Lens3:
        after_Compare_GlassList123.append([i + j])
#print(after_Compare_GlassList123)
#print(len(after_Compare_GlassList123))

for i in after_Compare_GlassList123:
    for j in new_GlassList_Lens45:
        after_Compare_GlassList.append([i + j])
#print(after_Compare_GlassList)
#print(len(after_Compare_GlassList))



print('\n--------Compare-END--------\n')