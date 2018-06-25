# 功能：提取XACML中规则(Rule)的5个关键属性Effect、Subject、Resourse、Action、Condition
# 算法：利用Python爬虫常用的BeautifulSoup像解析HTML一样解析XML，读取其对应标签的关键信息

from bs4 import BeautifulSoup  # 导入BeautifulSoup 模块

#xml_path = 'E:\PycharmProjects\XACML_Optimized_Matching\\'  # XML文件存放路径
#file_path = 'E:\PycharmProjects\XACML_Optimized_Matching\\'  # 提取后文件存放路径

#file = open(file_path + 'extraction.csv', 'w')  # 提取文件名extraction.csv
file = open('csv1_extraction.csv', 'w')

def extract():
    soup = BeautifulSoup(open( 'xacml3_asms.xml'), "lxml")

    #num = 0vl
    rule = soup.find_all('rule')
    for ru in rule:
        ef = ru.get('effect')
        file.write(ef + ',')  # effect写入到文件

        su = ru.subject.attributevalue.string
        if (su):  # 非空，则不变
            su
        else:  # 空，则赋值为空串
            su = ''
        file.write(su + ',')  # subject写入到文件

        re = ru.resource.attributevalue.string
        if (re):  # 非空，则不变
            re
        else:  # 空，则赋值为空串
            re = ''
        file.write(re + ',')  # resource写入到文件

        ac = ru.action.attributevalue.string
        if (ac):  # 非空，则不变
            ac
        else:  # 空，则赋值为空串
            ac = ''
        file.write(ac + ',')  # action写入到文件

        co = ru.condition.attributevalue.string
        if (co):  # 非空，则不变
            co
        else:  # 空，则赋值为空串
            co = ''
        file.write(co+ ',')  # condition写入到文件

        file.write('\n')

        #print(ef + ',' + su + ',' + re + ',' + ac + ',' + co)
        #num = num + 1

    #print(num)  # 打印测试一共遍历了多少条策略


extract()
