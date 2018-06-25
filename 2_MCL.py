import numpy as np
import random

filename='csv1_extraction.csv'
file=open(filename)

arrayOfLines=file.readlines()
#将文件中的内容每一行读取出来，保存在列表arrayOfLines中

size=320
#size表示测试用的策略集条数

def getLinesNumberForXml():#计算文件有多少行，返回文件行数
    lineOfnumber = 0
    for line in arrayOfLines:
        if '' != line:
            lineOfnumber += 1
    return lineOfnumber

def initMatrix():#初始化全0矩阵
    Matrix = np.zeros((size, size))
    np.dtype = 'float16'
    return Matrix

def creatAdjacencyMatrix(Matrix):#创建邻接矩阵
    # print(size)
    i=0
    while i < size:             #读取矩阵中横坐标为i的一条规则
        #line_i = i             #顺序读取第i行，如果需要笛卡尔积版的小规模策略集，顺序读取前n条就行，n为16倍数
        #如果顺序取前320条，则把上一行注释取消，把下一行注释掉
        line_i = arrayOfLines[random.randint(0, 8999)]   #随机读取0-8999行中的一行
        line_i = line_i.strip()  # 去掉首位空格
        listFromLine_i = line_i.split(',')  #split根据‘，’分割字符串，返回一个列表赋值给listFromLine_i
        j=0
        while j < size:             #读取矩阵中纵坐标为j的一条规则
            # line_j = j            #顺序读取第i行，如果需要笛卡尔积版的小规模策略集，顺序读取前n条就行，n为16倍数
            # 如果顺序取前320条，则把上一行注释取消，把下一行注释掉
            line_j = arrayOfLines[random.randint(0, 8999)]     #随机读取0-8999行中的一行
            line_j = line_j.strip()  # 去掉首位空格
            listFromLine_j = line_j.split(',')

            sign_S = 0
            sign_R = 0
            sign_A = 0
            sign_C = 0
            #用四位二进制数表是两条规则的关系，sign_S sign_R sign_A sign_C 的值为1表示对应的属性相同，0表示不同，初始化s r a c均不相同，均置为0

            if listFromLine_i[1] == listFromLine_j[1]:  # 如果规则i和规则j的Subject相同，将sign_S置1
                sign_S=1
            if listFromLine_i[2] == listFromLine_j[2]:  # 如果规则i和规则j的Resoure相同，将sign_S置1
                sign_R=1
            if listFromLine_i[3] == listFromLine_j[3]:  # 如果规则i和规则j的Action相同，将sign_S置1
                sign_A=1
            if listFromLine_i[4] == listFromLine_j[4]:  # 如果规则i和规则j的Condition相同，将sign_S置1
                sign_C = 1

            Matrix[i][j] = 8 * sign_S + 4 * sign_R + 2 * sign_A + 1 * sign_C  # 将矩阵的第i行j列的值设置为由四位二进制数对应的十进制数

            # print('i= '+str(i)+'j= '+str(j))
            #
            # print(Matrix[i][j])

            j += 1
        i += 1
    return Matrix

#markovCluster函数代码部分取自 https://blog.csdn.net/u010376788/article/details/50187321
def markovCluster(adjacencyMat, numIter, power=2, inflation=2):
    columnSum = np.sum(adjacencyMat, axis=0)    #一个一行的矩阵，每一列为邻接矩阵的该列所有值得和
    # print (columnSum)
    probabilityMat = adjacencyMat / columnSum   #标准化得矩阵，邻接矩阵得每一值除以该列所有值和
    # print(probabilityMat)

    # Expand by taking the e^th power of the matrix.
    def _expand(probabilityMat, power):
        expandMat = probabilityMat
        for i in range(power - 1):
            # print('i: '+str(i))
            expandMat = np.dot(expandMat, probabilityMat)
        return expandMat

    expandMat = _expand(probabilityMat, power)

    # Inflate by taking inflation of the resulting
    # matrix with parameter inflation.
    def _inflate(expandMat, inflation):
        powerMat = expandMat
        for i in range(inflation - 1):
            powerMat = powerMat * expandMat
        inflateColumnSum = np.sum(powerMat, axis=0)
        inflateMat = powerMat / inflateColumnSum
        return inflateMat

    inflateMat = _inflate(expandMat, inflation)

    # 训练的次数，320大约在12次收敛，数据稳定和前一次一样即可判断为收敛，
    # 比较不同训练次数的结果是否相同来判断是否收敛，次数不宜超过收敛的次数太多，
    for i in range(numIter):
        expand = _expand(inflateMat, power)
        inflateMat = _inflate(expand, inflation)


    # 由于NumPy的矩阵显示原因（只显示左上角，右上角，左下角，右下角的3*3部分，打印出其在中间未显示部分的非0值，）
    for i in range(0,size-1):
        for j in range(0,size-1):
            if inflateMat[i][j] != 0:
                print('i: ' + str(i) + ' j: ' + str(j) + ' valueOfMatrix:' + str(inflateMat[i][j]))
    print('训练结果')
    print(inflateMat)

if __name__ == '__main__':
    #初始化全0矩阵
    Matrix = initMatrix()

    #创建邻接矩阵
    Matrix = creatAdjacencyMatrix(Matrix)

    #可选，将矩阵保存下来，save保存的文件（npy格式）可直接在别得python文件中使用load加载
    # np.save('Matrix.npy',Matrix)

    # 可选，将矩阵保存下来，savetxt保存的文件为txt格式
    # np.savetxt('Matrix.txt', Matrix)

    # 可选，打印矩阵，现为邻接矩阵
    # print(Matrix)

    n=15       #训练次数
    markovCluster(Matrix, n)    #mcl聚类

    # 可选，将NumPY的matrix结构保存转化为字符串保存，发现只有左上角，右上角，左下角，右下角的3*3部分，
    # fileOfMatrix = 'txt1_Matrix.txt'
    # fOM = open(fileOfMatrix, 'w')
    # fOM.write(str(Matrix))
    # fOM.close()
    # file.close()

