#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.cross_decomposition import CCA as CCA_dd
from sklearn.datasets import make_biclusters
# from sklearn.datasets import samples_generator as sg
# from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn_extra.cluster import KMedoids
# from tslearn.clustering import silhouette_score as s_score
from sklearn.metrics import consensus_score
import numpy as np
# from matplotlib import pyplot as plt
from time import sleep
# import webbrowser as web
from sklearn.datasets import make_biclusters
from sklearn import manifold
from sklearn.cluster import SpectralCoclustering,KMeans,AgglomerativeClustering,SpectralClustering,Birch
from sklearn.metrics import consensus_score,silhouette_score,silhouette_samples
from flask import Flask,render_template,request,redirect,session,jsonify,url_for
from sklearn.cluster import SpectralCoclustering,SpectralBiclustering
from sklearn.decomposition import PCA as ppp
# from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import json
import time
import math
# from IPython.display import display
from scipy import stats
import re
import random
from sklearn import preprocessing
import os
import shutil

pathbase = os.path.dirname(os.path.realpath(__file__))

def maxminnorm(data):   #传入一个矩阵
    mins = data.min(0)  #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)  #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))  #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]      #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1)) #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData

dfg = pd.read_csv('static/data/gongl.csv')
datag = dfg.iloc[0:2000, [183,188]].values
# print(datag)
for i in range(datag.shape[0]):
    for j in range(datag.shape[1]):
        if math.isnan(datag[i,j]):
            datag[i,j]=datag[i-1,j]
toug = dfg.columns.values[[183,188]]

dff = pd.read_csv('static/data/biwen_data.csv')
dataa = dff.iloc[0:2000, 1:].values

zhud=dff.iloc[0:2000, 287:331].values
global cca_data
global cca_name

for i in range(dataa.shape[0]):
    for j in range(1,dataa.shape[1]):
        if math.isnan(dataa[i,j]):
            dataa[i,j]=dataa[i-1,j]
time = dff.iloc[0:2000, 1].values

pingjun=np.mean(dataa[:,1:],axis=0)

tou = dff.columns.values[1:]
for i in range(zhud.shape[0]):
    for j in range(zhud.shape[1]):
        if math.isnan(zhud[i,j]):
            # print([i,j])
            zhud[i,j]=zhud[i-1,j]
zhud=maxminnorm(zhud)
# print(zhud)
# print(np.corrcoef(zhud.T))
# print(np.corrcoef(zhud.T).shape)
# print(dff.columns.values[287:331])
# print(zhud.shape)
app=Flask(__name__)


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    filename = None
    if endpoint == 'static':
        filename = values.get('filename', None)
    if filename:
        file_path = os.path.join(app.root_path, endpoint, filename)
        values['v'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

# 实现跳转到页面index1 即: 低温过热器上部屏出口段后墙横向 数据页面
@app.route('/tiaoindex1',methods=['GET','POST'])
def tiaoindex1():
    return render_template("index1.html")

# 实现跳转到页面index2 即: 低温过热器下部屏出口段后墙横向 数据页面
@app.route('/tiaoindex2',methods=['GET','POST'])
def tiaoindex2():
    return render_template("index2.html")

# 实现跳转到页面index3 即: 低温过热器下部屏出口段后墙横向 数据页面
@app.route('/tiaoindex3',methods=['GET','POST'])
def tiaoindex3():
    return render_template("index3.html")


# 实现跳转到页面test_index2 即: 高温过热器出口段后墙横向 数据页面
@app.route('/test_index2',methods=['GET','POST'])
def test_index2():
    return render_template("index.html")

@app.route('/')
def hello_world():
    # return redirect('/cedian')
    return render_template('test_indexxxx.html')
    # return render_template('index.html')

@app.route('/cedian',methods=['GET','POST'])
def cedian():
    global cca_data
    global cca_name
    dff = pd.read_csv('static/data/biwen_data.csv')
    dataa = dff.iloc[:, 1:].values
    for i in range(dataa.shape[0]):
        for j in range(1,dataa.shape[1]):
            if math.isnan(dataa[i,j]):
                dataa[i,j]=dataa[i-1,j]
    df = pd.read_csv('static/data/biwen_name.csv',encoding='gb2312')
    column_headers = df.columns.values
    column_headers=np.array(column_headers)
    choose_id=request.form.get('choose_id')
    gao=[]
    gao2=[]
    gaoj=[]
    di1=[]
    di1j=[]
    dii=[]
    di2=[]
    di2j=[]
    dii2=[]
    qian=[]
    hou=[]
    houj=[]
    houu=[]
    zuo=[]
    you=[]
    for i in range(len(column_headers)):
        if '高温过热器' in column_headers[i]:
            gao.append(column_headers[i])
            gaoj.append(i)
            # print(np.round(np.mean(dataa[:,i]),2))
            # gao2.append(600+random.randint(0,9)+0.11)
            shuzi=np.round(np.mean(dataa[:,i]),2)
            # print(shuzi)
            # if np.isnan(shuzi):
            #     print(i)
            #     print(column_headers[i])
            gao2.append(shuzi)
        elif '低温过热器上部屏出口段后墙' in column_headers[i]:
            di1.append(column_headers[i])
            di1j.append(i)
            shuzi=np.round(np.mean(dataa[:,i]),2)
            dii.append(shuzi)
        elif '低温过热器下部屏出口段后墙' in column_headers[i]:
            di2.append(column_headers[i])
            di2j.append(i)
            shuzi=np.round(np.mean(dataa[:,i]),2)
            dii2.append(shuzi)
        elif '前墙' in column_headers[i]:
            qian.append(column_headers[i])
        elif '后墙隔墙' in column_headers[i]:
            hou.append(column_headers[i])
            houj.append(i)
            shuzi=np.round(np.mean(dataa[:,i]),2)
            houu.append(shuzi)
        elif '左墙' in column_headers[i]:
            zuo.append(column_headers[i])
        elif '右墙' in column_headers[i]:
            you.append(column_headers[i])
        else:
            print(column_headers[i])
    # print(len(gao),gao)
    # print(gao[141])
    # print(len(di1),di1)
    # print(len(di2),di2)
    # print(len(qian),qian)
    # print(len(hou),hou)
    # print(len(zuo),zuo)
    # print(len(you),you)
    ced=[]
    ced2=[]
    ced3=[]
    ced4=[]
    for i in range(1,83):
        pai='第'+str(i)+'排'
        for j in range(len(gao)):
            if pai in gao[j]:
                for k in range(1,9):
                    guan=str(k)+'管'
                    if guan in gao[j]:
                        # print("s",gao[j])
                        # print(guan)
                        # print(pai+guan)
                        cedi=[i,0.6,k,'高温过热器出口段横向'+pai+'#'+guan,gao2[j]]
                        ced.extend([cedi])

    for i in range(1,21):
        pai='第'+str(i)+'排'
        for j in range(len(di1)):
            if pai in di1[j]:
                for k in range(1,14):
                    guan='#'+str(k)+'管'
                    if guan in di1[j]:
                        # print(di1[j],i,j,k)
                        # print(pai+guan)
                        cedi=[i,0.6,k,'低温过热器上部屏出口段后墙横向'+pai+guan,dii[j]]
                        ced2.extend([cedi])
    # print(np.array(ced2).shape)
    for i in range(1,21):
        pai='第'+str(i)+'排'
        for j in range(len(di2)):
            if pai in di2[j]:
                for k in range(13,14):
                    guan=str(k)+'管'
                    if guan in di2[j]:
                        # print(pai+guan)
                        cedi=[i,0.6,k,'低温过热器下部屏出口段后墙横向'+pai+'#'+guan,dii2[j]]
                        ced3.extend([cedi])

    for i in range(1,325):
        pai='#'+str(i)+'管'
        guan=str(i)+'管'
        for j in range(len(hou)):
            if pai in hou[j]:
                cedi=[i,0.6,5,'后墙隔墙出口段#'+guan,houu[j]]
                ced4.extend([cedi])
    # c=[1,0.6,6,'none',houu[0]]
    # ced4.extend([c])
    #
    # c2=[1,0.6,14,'none',dii2[0]]
    # ced3.extend([c2])

    zhuz=[]
    zhuz2=[]
    zhuz3=[]
    zhuz4=[]
    julei_i1=[]
    julei_i2=[]
    julei_i3=[]
    julei_i4=[]
    lei1_d=[]
    lei2_d=[]
    lei3_d=[]
    lei4_d=[]
    lei_d=[]
    julei_n1=[]
    julei_n2=[]
    julei_n3=[]
    julei_n4=[]
    all_xgx=[]
    for i in range(1,83):
        zhuzi=[i,1,9]
        zhuz.append(zhuzi)
    for i in range(1,21):
        zhuzi=[i,1,14]
        zhuz2.append(zhuzi)
    for i in range(1,21):
        zhuzi=[i,1,14]
        zhuz3.append(zhuzi)
    for i in range(1,324):
        zhuzi=[i,1,14]
        zhuz4.append(zhuzi)
    # print(zhuz4)
    # print(zhuz)
    # print(ced4)
    # print(ced)
    # print(max(gao2))
    # print(min(gao2))
    # print("choose_id",choose_id)
    biwen_col=gao
    for i in range(len(biwen_col)):
        biwen_col[i]=biwen_col[i].replace('壁温','');

    # 低上
    biwen_col1 = di1
    for i in range(len(biwen_col1)):
        biwen_col1[i] = biwen_col1[i].replace('壁温', '');
    # 低下
    biwen_col2 = di2
    for i in range(len(biwen_col2)):
        biwen_col2[i] = biwen_col2[i].replace('壁温', '');
    # 后墙
    biwen_col3 = hou
    for i in range(len(biwen_col3)):
        biwen_col3[i] = biwen_col3[i].replace('壁温', '');
    # print(biwen_col)
    qita = pd.read_csv("static/data/gongl.csv").iloc[:, 156:].values
    for i in range(qita.shape[0]):
        for j in range(qita.shape[1]):
            if math.isnan(qita[i,j]):
                qita[i,j]=qita[i-1,j]
    qita_c=pd.read_csv("static/data/gongl.csv").columns.values[156:]
    # r_y_d=[]
    r_y_n=['低过上后聚类融合1','低过上后聚类融合2','低过上后公因子','低过下后聚类融合1','低过下后聚类融合2','低过上后公因子','高过聚类融合1','高过聚类融合2','高过聚类融合3','高过聚类融合4','高温过热器公因子1','高温过热器公因子2','高温过热器公因子3','后墙隔墙聚类融合1','后墙隔墙聚类融合2','后墙隔墙聚类融合3','后墙隔墙公因子1','后墙隔墙公因子2']
    r_y_n=r_y_n+list(qita_c)
    dsh_ju1 = pd.read_csv("static/data/低过上后聚类融合1.csv").iloc[:, 1].values.reshape((-1,1))
    dsh_ju2 = pd.read_csv("static/data/低过上后聚类融合2.csv").iloc[:, 1].values.reshape((-1,1))
    dsh_y = pd.read_csv("static/data/低过上后因子分析.csv").iloc[:, 0].values.reshape((-1,1))
    dxh_ju1 = pd.read_csv("static/data/低过下后聚类融合1.csv").iloc[:, 2].values.reshape((-1,1))
    dxh_ju2 = pd.read_csv("static/data/低过下后聚类融合2.csv").iloc[:, 2].values.reshape((-1,1))
    dxh_y = pd.read_csv("static/data/低过下后因子分析.csv").iloc[:, 0].values.reshape((-1,1))
    gaoju1 = pd.read_csv("static/data/高过聚类融合1.csv").iloc[:, 1].values.reshape((-1,1))
    gaoju2 = pd.read_csv("static/data/高过聚类融合2.csv").iloc[:, 1].values.reshape((-1,1))
    gaoju3 = pd.read_csv("static/data/高过聚类融合3.csv").iloc[:, 1].values.reshape((-1,1))
    gaoju4 = pd.read_csv("static/data/高过聚类融合4.csv").iloc[:, 1].values.reshape((-1,1))
    gao_y = pd.read_csv("static/data/高温过热器因子分析.csv").iloc[:, 0:3].values
    houju1 = pd.read_csv("static/data/后墙隔墙聚类融合1.csv").iloc[:, 1].values.reshape((-1,1))
    houju2 = pd.read_csv("static/data/后墙隔墙聚类融合2.csv").iloc[:, 1].values.reshape((-1,1))
    houju3 = pd.read_csv("static/data/后墙隔墙聚类融合3.csv").iloc[:, 1].values.reshape((-1,1))
    hou_y = pd.read_csv("static/data/后墙隔墙因子分析.csv").iloc[:, 0:2].values
    r_y_d=np.hstack((dsh_ju1,dsh_ju2))
    r_y_d=np.hstack((r_y_d,dsh_y))
    r_y_d=np.hstack((r_y_d,dxh_ju1))
    r_y_d=np.hstack((r_y_d,dxh_ju2))
    r_y_d=np.hstack((r_y_d,dxh_y))
    r_y_d=np.hstack((r_y_d,gaoju1))
    r_y_d=np.hstack((r_y_d,gaoju2))
    r_y_d=np.hstack((r_y_d,gaoju3))
    r_y_d=np.hstack((r_y_d,gaoju4))
    r_y_d=np.hstack((r_y_d,gao_y))
    r_y_d=np.hstack((r_y_d,houju1))
    r_y_d=np.hstack((r_y_d,houju2))
    r_y_d=np.hstack((r_y_d,houju3))
    r_y_d=np.hstack((r_y_d,hou_y))
    r_y_d=np.hstack((r_y_d,qita))
    # print(r_y_d)
    for i in range(r_y_d.shape[0]):
        for j in range(r_y_d.shape[1]):
            if math.isnan(r_y_d[i,j]):
                r_y_d[i,j]=r_y_d[i-1,j]
    r_y_corr=np.corrcoef(r_y_d.T)
    cca_data=r_y_d.copy()
    cca_name=r_y_n
    # print("r_y_d.shape",r_y_d.shape)
    # print("r_y_corr.shape",r_y_corr.shape)
    if choose_id=='高温过热器出口段横向':
        ma=max(gao2)
        mi=min(gao2)
        data1=ced
        data2=zhuz
        cedianxx=dataa[:,gaoj]
        name=gao
        lei = pd.read_csv('static/data/高过4聚类.csv')
        leibie = lei.iloc[:, 0].values
        leibie=leibie.reshape((-1,1))
        for i in range(leibie.shape[0]):
            if leibie[i]==0:
                julei_i1.append(i)
                julei_n1.append(gao[i])
            elif leibie[i]==1:
                julei_i2.append(i)
                julei_n2.append(gao[i])
            elif leibie[i]==2:
                julei_i3.append(i)
                julei_n3.append(gao[i])
            elif leibie[i]==3:
                julei_i4.append(i)
                julei_n4.append(gao[i])
        julei_d1=cedianxx[:,julei_i1]
        julei_d2=cedianxx[:,julei_i2]
        julei_d3=cedianxx[:,julei_i3]
        julei_d4=cedianxx[:,julei_i4]
        # print(len(julei_i1),len(julei_i2),len(julei_i3),len(julei_i4))
        # print("shape",julei_d1.shape)
        data1=np.array(data1)
        # print(data1.shape)
        data1=np.hstack((data1,leibie.reshape(-1,1)))
        # qita = pd.read_csv("static/data/gongl.csv").iloc[:, 156:].values
        # qita_c=pd.read_csv("static/data/gongl.csv").columns.values[156:]
        # print("qita",qita.shape)
        all_data=(np.hstack((cedianxx,qita))).astype(float)
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                if math.isnan(all_data[i,j]):
                    all_data[i,j]=all_data[i-1,j]
        all_xgx=np.corrcoef(all_data.T)
        julei_dd=np.hstack((julei_d1,julei_d2))
        julei_dd=np.hstack((julei_dd,julei_d3))
        julei_dd=np.hstack((julei_dd,julei_d4))
        julei_dd=np.hstack((julei_dd,qita))
        for i in range(julei_dd.shape[0]):
            for j in range(julei_dd.shape[1]):
                if math.isnan(julei_dd[i,j]):
                    julei_dd[i,j]=julei_dd[i-1,j]
        julei_dd = julei_dd.astype(float)
        dange_corr=np.corrcoef(julei_dd.T)
        dange_corr=np.trunc(dange_corr*100)
        # print("cor",dange_corr)
        # for i in range(dange_corr.shape[0]):
        #     for j in range(dange_corr.shape[1]):
        #         if math.isnan(dange_corr[i,j]):
                    # print(dange_corr[i,j])
                    # print(i,j)
        lei1_corr=dange_corr[0:len(julei_i1),len(gaoj):]
        lei2_corr=dange_corr[len(julei_i1):len(julei_i1)+len(julei_i2),len(gaoj):]
        lei3_corr=dange_corr[len(julei_i1)+len(julei_i2):len(julei_i1)+len(julei_i2)+len(julei_i3),len(gaoj):]
        lei4_corr=dange_corr[len(julei_i1)+len(julei_i2)+len(julei_i3):len(julei_i1)+len(julei_i2)+len(julei_i3)+len(julei_i4),len(gaoj):]
        # lei1_d=[]
        # lei2_d=[]
        # lei3_d=[]
        # lei4_d=[]
        for i in range(lei1_corr.shape[0]):
            for j in range(lei1_corr.shape[1]):
                cedii=[i,j,lei1_corr[i,j],qita_c[j],biwen_col[julei_i1[i]]]
                lei1_d.extend([cedii])
        for i in range(lei2_corr.shape[0]):
            for j in range(lei2_corr.shape[1]):
                cedii=[i,j,lei2_corr[i,j],qita_c[j],biwen_col[julei_i2[i]]]
                lei2_d.extend([cedii])
        for i in range(lei3_corr.shape[0]):
            for j in range(lei3_corr.shape[1]):
                cedii=[i,j,lei3_corr[i,j],qita_c[j],biwen_col[julei_i3[i]]]
                lei3_d.extend([cedii])
        for i in range(lei4_corr.shape[0]):
            for j in range(lei4_corr.shape[1]):
                cedii=[i,j,lei4_corr[i,j],qita_c[j],biwen_col[julei_i4[i]]]
                lei4_d.extend([cedii])
        lei_d.extend(([lei1_d]))
        lei_d.extend(([lei2_d]))
        lei_d.extend(([lei3_d]))
        lei_d.extend(([lei4_d]))
        # print(lei1_corr.shape)
        # print(lei1_corr)
        # print(lei2_corr.shape)
        # print(lei2_corr)
        # qita_c=pd.read_csv("static/data/gongl.csv").columns.values[156:]
        dange_name=gao+list(qita_c)
        # print(dange_name)
        # print(len(dange_name))
        # print(dange_name[509])
        # gaoju1 = pd.read_csv("static/data/高过聚类融合1.csv").iloc[:, 1].values.reshape((-1,1))
        # gaoju2 = pd.read_csv("static/data/高过聚类融合2.csv").iloc[:, 1].values.reshape((-1,1))
        # gaoju3 = pd.read_csv("static/data/高过聚类融合3.csv").iloc[:, 1].values.reshape((-1,1))
        # gaoju4 = pd.read_csv("static/data/高过聚类融合4.csv").iloc[:, 1].values.reshape((-1,1))
        julei_d=np.hstack((gaoju1,gaoju2))
        julei_d=np.hstack((julei_d,gaoju3))
        julei_d=np.hstack((julei_d,gaoju4))
        # print(julei_d.shape,qita.shape)
        julei_d=np.hstack((julei_d,qita))
        for i in range(julei_d.shape[0]):
            for j in range(julei_d.shape[1]):
                if math.isnan(julei_d[i,j]):
                    julei_d[i,j]=julei_d[i-1,j]
        julei_d = julei_d.astype(float)
        rong_corr=np.corrcoef(julei_d.T)
        # print(rong_corr)
        # for i in range(rong_corr.shape[0]):
        #     for j in range(rong_corr.shape[1]):
        #         if math.isnan(rong_corr[i,j]):
        #             print(rong_corr[i,j])
        #             print(i,j)
        # print(julei_d[:,167])
        xgju=rong_corr[0,:].reshape((1,-1))
        xgju=np.delete(xgju,[0,1,2,3],axis=1)
        for i in range(1,4):
            # print(i)
            xgju=np.vstack((xgju,np.delete(rong_corr[i,:].reshape((1,-1)),[0,1,2,3],axis=1)))
        # print(xgju)
        # print(xgju.shape)
        # print("ronghe",rong_corr.shape)
    elif choose_id=='低温过热器上部屏出口段后墙横向':
        ma=max(dii)
        mi=min(dii)
        data1=ced2
        data2=zhuz2
        cedianxx=dataa[:,di1j]
        name=di1
        lei = pd.read_csv('jw/低过上后2聚类.csv')
        leibie = lei.iloc[:, 0].values
        leibie=leibie.reshape((-1,1))
        data1=np.array(data1)
        data1=np.hstack((data1,leibie.reshape(-1,1)))
        for i in range(leibie.shape[0]):
            if leibie[i]==0:
                julei_i1.append(i)               # 将聚类后类别为0的下标值添加到列表中
                julei_n1.append(di1[i])          # 将类别为0 具体的 低温过热器ID 添加到列表中
            elif leibie[i]==1:
                julei_i2.append(i)
                julei_n2.append(di1[i])

        # 将聚类后的 各测点的壁温温度添加到相应列表中
        julei_d1=cedianxx[:,julei_i1]
        julei_d2=cedianxx[:,julei_i2]
        data1=np.array(data1)
        data1=np.hstack((data1,leibie.reshape(-1,1)))
        all_data=(np.hstack((cedianxx,qita))).astype(float)
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                if math.isnan(all_data[i,j]):
                    all_data[i,j]=all_data[i-1,j]
        all_xgx=np.corrcoef(all_data.T)
        julei_dd=np.hstack((julei_d1,julei_d2))
        julei_dd=np.hstack((julei_dd,qita))
        for i in range(julei_dd.shape[0]):
            for j in range(julei_dd.shape[1]):
                if math.isnan(julei_dd[i,j]):
                    julei_dd[i,j]=julei_dd[i-1,j]
        julei_dd = julei_dd.astype(float)
        dange_corr=np.corrcoef(julei_dd.T)
        # 舍弃小数部分并乘上100
        dange_corr=np.trunc(dange_corr*100)
        # for i in range(dange_corr.shape[0]):
        #     for j in range(dange_corr.shape[1]):
        #         if math.isnan(dange_corr[i,j]):
        #             print(dange_corr[i,j])
        #             print(i,j)
        # 分别计算 每一个聚类后的类别与参数之间的相关性
        lei1_corr=dange_corr[0:len(julei_i1),len(di1j):]
        lei2_corr=dange_corr[len(julei_i1):len(julei_i1)+len(julei_i2),len(di1j):]
        for i in range(lei1_corr.shape[0]):
            for j in range(lei1_corr.shape[1]):
                cedii=[i,j,lei1_corr[i,j],qita_c[j],biwen_col1[julei_i1[i]]]
                lei1_d.extend([cedii])
        # 计算聚类类别 1 和 参数之间的相关性
        for i in range(lei2_corr.shape[0]):
            for j in range(lei2_corr.shape[1]):
                cedii=[i,j,lei2_corr[i,j],qita_c[j],biwen_col1[julei_i2[i]]]
                lei2_d.extend([cedii])
        # lei_d 存储的是 所有类别与参数的相关性信息
        lei_d.extend(([lei1_d]))
        lei_d.extend(([lei2_d]))
        # print("总的相关性信息 lei_d:")
        # print(lei_d)
        # print("+++")
        dange_name=gao+list(qita_c)
        # 一个融合类别与参数的相关性
        gaoju1 = pd.read_csv("static/data/低过上后聚类融合1.csv").iloc[:, 1].values.reshape((-1,1))
        gaoju2 = pd.read_csv("static/data/低过上后聚类融合2.csv").iloc[:, 1].values.reshape((-1,1))
        julei_d=np.hstack((gaoju1,gaoju2))

        # print(julei_d.shape,qita.shape)
        julei_d=np.hstack((julei_d,qita))
        for i in range(julei_d.shape[0]):
            for j in range(julei_d.shape[1]):
                if math.isnan(julei_d[i,j]):
                    julei_d[i,j]=julei_d[i-1,j]
        julei_d = julei_d.astype(float)
        rong_corr=np.corrcoef(julei_d.T)

        xgju=rong_corr[0,:].reshape((1,-1))
        xgju=np.delete(xgju,[0,1],axis=1)
        for i in range(1,2):
            # print(i)
            xgju=np.vstack((xgju,np.delete(rong_corr[i,:].reshape((1,-1)),[0,1],axis=1)))
        # print("sgju 如下:")
        # print(xgju)
        # print(xgju.shape)
        # print("ronghe",rong_corr.shape)
    elif choose_id=='低温过热器下部屏出口段后墙横向':
        ma=max(dii2)
        mi=min(dii2)
        data1=ced3
        data2=zhuz3
        cedianxx=dataa[:,di2j]
        name=di2
        lei = pd.read_csv('static/data/低过下后2聚类.csv')
        leibie = lei.iloc[:, 0].values
        leibie=leibie.reshape((-1,1))
        # print(leibie.shape)
        data1=np.array(data1)
        # print(data1.shape)
        # print(data1)
        data1=np.hstack((data1,leibie.reshape(-1,1)))
        # 循环遍历 聚类序号的列表(低温过热器只有类别 0 1)
        for i in range(leibie.shape[0]):
            if leibie[i]==0:
                julei_i1.append(i)               # 将聚类后类别为0的下标值添加到列表中
                julei_n1.append(di2[i])          # 将类别为0 具体的 低温过热器ID 添加到列表中
            elif leibie[i]==1:
                julei_i2.append(i)
                julei_n2.append(di2[i])

        # 将聚类后的 各测点的壁温温度添加到相应列表中
        julei_d1=cedianxx[:,julei_i1]
        julei_d2=cedianxx[:,julei_i2]


        # print("la下lalalalala")
        # print(julei_i1)                 # 打印下标
        # print(julei_n1)                 # 打印低温过热器对应ID
        # print(julei_d1)                 # 打印对应ID的温度
        # print("la下lalalalala")

        # 打印低温过热器聚类后(0,1)两类别的长度
        # print(len(julei_i1),len(julei_i2))
        # # 打印测点列表对应的长和宽 5797行 252列
        # print("shape",julei_d1.shape)

        data1=np.array(data1)
        # print(data1)
        # print(data1.shape)
        data1=np.hstack((data1,leibie.reshape(-1,1)))
        all_data=(np.hstack((cedianxx,qita))).astype(float)
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                if math.isnan(all_data[i,j]):
                    all_data[i,j]=all_data[i-1,j]
        all_xgx=np.corrcoef(all_data.T)

        # 将测点温度矩阵和参数 进行 行连接形成总矩阵
        julei_dd=np.hstack((julei_d1,julei_d2))
        julei_dd=np.hstack((julei_dd,qita))
        # print("将两个测点矩阵进行 行连接后的总矩阵为:")
        # print(julei_dd)

        # 对总矩阵进行缺失值处理
        for i in range(julei_dd.shape[0]):
            for j in range(julei_dd.shape[1]):
                if math.isnan(julei_dd[i,j]):
                    julei_dd[i,j]=julei_dd[i-1,j]
        julei_dd = julei_dd.astype(float)

        # 求皮尔逊相关性系数
        dange_corr=np.corrcoef(julei_dd.T)
        # 舍弃小数部分并乘上100
        dange_corr=np.trunc(dange_corr*100)
        # print("cor",dange_corr)
        # 对相关性矩阵进行缺失值处理
        # for i in range(dange_corr.shape[0]):
        #     for j in range(dange_corr.shape[1]):
        #         if math.isnan(dange_corr[i,j]):
        #             print(dange_corr[i,j])
        #             print(i,j)

        # 分别计算 每一个聚类后的类别与参数之间的相关性
        lei1_corr=dange_corr[0:len(julei_i1),len(di2j):]
        lei2_corr=dange_corr[len(julei_i1):len(julei_i1)+len(julei_i2),len(di2j):]

        # print("lei1_corr:")
        # print(lei1_corr)

        # 计算聚类类别 0 和 参数之间的相关性
        for i in range(lei1_corr.shape[0]):
            for j in range(lei1_corr.shape[1]):
                cedii=[i,j,lei1_corr[i,j],qita_c[j],biwen_col2[julei_i1[i]]]
                lei1_d.extend([cedii])

        # 打印聚类后类别 0 的各温度测点和参数的相关性
        # print("lei1_d")
        # print(lei1_d)

        # 计算聚类类别 1 和 参数之间的相关性
        for i in range(lei2_corr.shape[0]):
            for j in range(lei2_corr.shape[1]):
                cedii=[i,j,lei2_corr[i,j],qita_c[j],biwen_col2[julei_i2[i]]]
                lei2_d.extend([cedii])

        # lei_d 存储的是 所有类别与参数的相关性信息
        lei_d.extend(([lei1_d]))
        lei_d.extend(([lei2_d]))
        # print("总的相关性信息 lei_d:")
        # print(lei_d)
        # print("+++")
        dange_name=gao+list(qita_c)

        # 一个融合类别与参数的相关性
        gaoju1 = pd.read_csv("static/data/低过下后聚类融合1.csv").iloc[:, 2].values.reshape((-1,1))
        gaoju2 = pd.read_csv("static/data/低过下后聚类融合2.csv").iloc[:, 2].values.reshape((-1,1))
        julei_d=np.hstack((gaoju1,gaoju2))

        # print(julei_d.shape,qita.shape)
        julei_d=np.hstack((julei_d,qita))
        for i in range(julei_d.shape[0]):
            for j in range(julei_d.shape[1]):
                if math.isnan(julei_d[i,j]):
                    julei_d[i,j]=julei_d[i-1,j]

        julei_d = julei_d.astype(float)
        # print("julei_d")
        # print(julei_d)

        # # 低温下部里数据有空值 第二列为空值 打印空值
        # print("空值如下:")
        # print(julei_d[:, 1])
        # # 将这一列空值进行赋值
        # print("处理空值之后如下:")
        # julei_d[:,1] = julei_d[:,2]
        # print(julei_d[:, 1])
        # 再次打印
        # datajuleid = pd.DataFrame(julei_d)
        # datajuleid.to_csv("juleid.csv")

        rong_corr=np.corrcoef(julei_d.T)

        xgju=rong_corr[0,:].reshape((1,-1))
        xgju=np.delete(xgju,[0,1],axis=1)
        for i in range(1,2):
            # print(i)
            xgju=np.vstack((xgju,np.delete(rong_corr[i,:].reshape((1,-1)),[0,1],axis=1)))
        # print("sgju 如下:")
        # print(xgju)
        # print(xgju.shape)
        # print("ronghe",rong_corr.shape)
    elif choose_id=='后墙隔墙出口段':
        ma=max(houu)
        mi=min(houu)
        data1=ced4
        data2=zhuz4
        cedianxx=dataa[:,houj]
        name=hou
        lei = pd.read_csv('static/data/后墙隔墙3聚类.csv')
        leibie = lei.iloc[:, 0].values
        leibie=leibie.reshape((-1,1))
        data1=np.array(data1)
        data1=np.hstack((data1,leibie.reshape(-1,1)))
        # 循环遍历 聚类序号的列表
        for i in range(leibie.shape[0]):
            if leibie[i]==0:
                julei_i1.append(i)               # 将聚类后类别为0的下标值添加到列表中
                julei_n1.append(hou[i])          # 将类别为0 具体的 高温过热器ID 添加到列表中
            elif leibie[i]==1:
                julei_i2.append(i)
                julei_n2.append(hou[i])
            elif leibie[i]==2:
                julei_i3.append(i)
                julei_n3.append(hou[i])

        # 将聚类后的 各测点的壁温温度添加到相应列表中
        julei_d1=cedianxx[:,julei_i1]
        julei_d2=cedianxx[:,julei_i2]
        julei_d3=cedianxx[:,julei_i3]

        # print("la后lalala")
        # print(julei_i1)                 # 打印下标
        # print(julei_n1)                 # 打印高温过热器对应ID
        # print(julei_d1)                 # 打印对应ID的温度
        # print("la后lalala")

        # print(len(julei_i1),len(julei_i2),len(julei_i3))
        # # 打印测点列表对应的长和宽 5797行 252列
        # print("shape",julei_d1.shape)

        data1=np.array(data1)
        # print(data1)
        # print(data1.shape)
        data1=np.hstack((data1,leibie.reshape(-1,1)))

        # 可能影响壁温的参数的值
        qita = pd.read_csv("static/data/gongl.csv").iloc[:, 156:].values
        # 可能影响壁温参数的ID
        qita_c=pd.read_csv("static/data/gongl.csv").columns.values[156:]
        # print("可能影响壁温参数列表shape:")
        # print("qita",qita.shape)

        all_data=(np.hstack((cedianxx,qita))).astype(float)
        for i in range(all_data.shape[0]):
            for j in range(all_data.shape[1]):
                if math.isnan(all_data[i,j]):
                    all_data[i,j]=all_data[i-1,j]
        all_xgx=np.corrcoef(all_data.T)

        # 将测点温度矩阵和参数 进行 行连接形成总矩阵
        julei_dd=np.hstack((julei_d1,julei_d2))
        julei_dd=np.hstack((julei_dd,julei_d3))
        julei_dd=np.hstack((julei_dd,qita))
        # print("将三个测点矩阵进行 行连接后的总矩阵为:")
        # print(julei_dd)

        # 对总矩阵进行缺失值处理
        for i in range(julei_dd.shape[0]):
            for j in range(julei_dd.shape[1]):
                if math.isnan(julei_dd[i,j]):
                    julei_dd[i,j]=julei_dd[i-1,j]
        julei_dd = julei_dd.astype(float)

        # 求皮尔逊相关性系数
        dange_corr=np.corrcoef(julei_dd.T)
        # 舍弃小数部分并乘上100
        dange_corr=np.trunc(dange_corr*100)
        # print("cor",dange_corr)
        # 对相关性矩阵进行缺失值处理
        # for i in range(dange_corr.shape[0]):
        #     for j in range(dange_corr.shape[1]):
        #         if math.isnan(dange_corr[i,j]):
        #             print(dange_corr[i,j])
        #             print(i,j)

        # 分别计算 每一个聚类后的类别与参数之间的相关性
        lei1_corr=dange_corr[0:len(julei_i1),len(houj):]
        lei2_corr=dange_corr[len(julei_i1):len(julei_i1)+len(julei_i2),len(houj):]
        lei3_corr=dange_corr[len(julei_i1)+len(julei_i2):len(julei_i1)+len(julei_i2)+len(julei_i3),len(houj):]

        # print("lei1_corr:")
        # print(lei1_corr)

        for i in range(lei1_corr.shape[0]):
            for j in range(lei1_corr.shape[1]):
                cedii=[i,j,lei1_corr[i,j],qita_c[j],biwen_col3[julei_i1[i]]]
                lei1_d.extend([cedii])
        # 打印聚类后类别0 的各温度测点和参数的相关性
        # print("lei1_d")
        # print(lei1_d)
        for i in range(lei2_corr.shape[0]):
            for j in range(lei2_corr.shape[1]):
                cedii=[i,j,lei2_corr[i,j],qita_c[j],biwen_col3[julei_i2[i]]]
                lei2_d.extend([cedii])
        for i in range(lei3_corr.shape[0]):
            for j in range(lei3_corr.shape[1]):
                cedii=[i,j,lei3_corr[i,j],qita_c[j],biwen_col3[julei_i3[i]]]
                lei3_d.extend([cedii])


        # lei_d 存储的是 所有类别与参数的相关性信息
        lei_d.extend(([lei1_d]))
        lei_d.extend(([lei2_d]))
        lei_d.extend(([lei3_d]))

        dange_name=gao+list(qita_c)

        # 一个融合类别与参数的相关性
        gaoju1 = pd.read_csv("static/data/后墙隔墙聚类融合1.csv").iloc[:, 1].values.reshape((-1,1))
        gaoju2 = pd.read_csv("static/data/后墙隔墙聚类融合2.csv").iloc[:, 1].values.reshape((-1,1))
        gaoju3 = pd.read_csv("static/data/后墙隔墙聚类融合3.csv").iloc[:, 1].values.reshape((-1,1))

        julei_d=np.hstack((gaoju1,gaoju2))
        julei_d=np.hstack((julei_d,gaoju3))

        # print(julei_d.shape,qita.shape)
        julei_d=np.hstack((julei_d,qita))
        for i in range(julei_d.shape[0]):
            for j in range(julei_d.shape[1]):
                if math.isnan(julei_d[i,j]):
                    julei_d[i,j]=julei_d[i-1,j]
        julei_d = julei_d.astype(float)
        rong_corr=np.corrcoef(julei_d.T)

        xgju=rong_corr[0,:].reshape((1,-1))
        xgju=np.delete(xgju,[0,1,2],axis=1)
        for i in range(1,3):
            # print(i)
            xgju=np.vstack((xgju,np.delete(rong_corr[i,:].reshape((1,-1)),[0,1,2],axis=1)))
        # print(xgju)
        # print(xgju.shape)
        # print("ronghe",rong_corr.shape)
    # cedianx=np.corrcoef(cedianxx)
    data3=cedianxx
    cedianxx=np.array(cedianxx.tolist())
    cedianx=np.corrcoef(cedianxx.T)
    cedianx=np.round(cedianx,3)
    # print(gaoj)
    # print(cedianxx)
    # print("cedianxx",cedianxx.T.shape)
    # print(cedianx.shape)
    # print(cedianx)
    xgx=[]
    for i in range(cedianx.shape[0]):
        for j in range(cedianx.shape[0]):
            if i>j:
                n=[i,j,cedianx[i,j],name[i],name[j]]
                xgx.extend([n])
    # print(xgx)
    # for i in range(data3.shape[1]):
    #     if data3[0,i]==577.1709999999998:
    #         print("22",i)
    #         print(gao[i])
    # biwen_col=gao
    # for i in range(len(biwen_col)):
    #     biwen_col[i]=biwen_col[i].replace('壁温','');
    # print(biwen_col)
    return jsonify({
        "data1": data1.tolist(),
        "data2": data2,
        "max": ma,
        "min": mi,
        "xgx":xgx,
        "lei":leibie.tolist(),
        "data3":data3[:,:].tolist(),
        "biwen_col":biwen_col,
        "time":dataa[:,0].tolist(),
        "xgx":xgju.tolist(),
        "xgx_col":qita_c.tolist(),
        "yx_data":r_y_d.T.tolist(),
        "yx_name":r_y_n,
        "yx_corr":r_y_corr.tolist(),
        "zb_d":lei_d,
        "zb_nx":julei_n1,
        "all_xgx":all_xgx.tolist()
    })

@app.route('/zhexian',methods=['GET','POST'])
def zhexian():
    # print(request.form)
    # print(request.form.get)
    choose_id=request.form.get('choose_id')
    choose_id=json.loads(choose_id)
    # print(choose_id,len(choose_id))
    a=[]
    for j in choose_id:
        # print(j)
        # print(tou)
        for i in range(tou.shape[0]):
            if j in tou[i]:
                a.append(i)
    # print("a",a)
    # print(dataa.shape)
    data=dataa[:,a]
    # print(data)
    # print(data.shape)
    data = data.astype(float)
    data=np.round(data,2)
    # print("len(choose_id)",len(choose_id))
    # data=data.reshape(len(choose_id),-1)
    data=data.T
    # data=np.round(data,1)
    # print(data.shape)
    data=data[0:2000,:].tolist()
    # print(data)

    x_axis_data = [i for i in range(1000)]
    return jsonify({
        "data1": data,
        "xData":time.tolist()
    })
# 获取前端选择框选择的 index 下标值 并返回均值到前端
@app.route('/xuanqu',methods=['GET','POST'])
def xuanqu():
    wenjian=request.form.get('wenjian')
    # print(wenjian)
    df = pd.read_csv('static/data/工况/'+wenjian+'/bw.csv')
    # df = pd.read_csv("static/data/biwen_data.csv")
    df_name = pd.read_csv('static/data/biwen_name.csv', encoding='gb2312')
    column_headers = df.columns.values
    column_headers = np.array(column_headers)

    print("所选功率文件集:",wenjian)

    # dff1只为读取对应的时间
    dff1 = pd.read_csv("./static/data/工况/"+ wenjian + '/bw.csv')
    dataa = dff1.iloc[:, 0].values
    time = dataa.tolist()

    timeb = time[0]
    timeo = time[-1]

    timenew = []
    for i in range(len(time)):
        a = time[i]
        # print("---",a)
        timenew.append(a[10:])

    choose_id = request.form.get('choose_id')
    # print("选取框时的ID:",choose_id)
    indexKaung = request.form.get("index")
    indexKaung = json.loads(indexKaung)

    # print("indexkuang",indexKaung)


    # erced 最后一个值表示 当前ID的全部数据的平均值
    erced,ma,mi = cedianwz(choose_id,wenjian)

    # 获取选取ID的值
    xuanquID = []
    for i in range(len(indexKaung)):
        # print(indexKaung[i])
        # print(erced[indexKaung[i]])
        print("选取的ID:",erced[indexKaung[i]][2])
        xuanquID.append(erced[indexKaung[i]][2])
    # print(xuanquID)

    # 获取index 对应的 ID 坐标
    mId = []
    for j in range(len(xuanquID)):
        for i in range(len(column_headers)):
            if (xuanquID[j] in column_headers[i]):
                mId.append(i)

    # print("mId",mId)
    data = df.iloc[:, mId].values
    # print(data)

    # 获取每一行的均值
    # print("每一行均值:")
    datajun = np.mean(data, axis=1)
    # print(datajun)
    # print(datajun.tolist())

    return jsonify({
        "datajun" : datajun.tolist(),
        "time":timenew,        # 前2000行时间
        "timeb":timeb,
        "timeo":timeo
    })


# 壁温刷选事件 + 时间托选框
@app.route('/xuanqu2',methods=['GET','POST'])
def xuanqu2():
    # df = pd.read_csv("static/data/biwen_data.csv")
    wenjian=request.form.get('wenjian')
    print(wenjian)
    df = pd.read_csv('static/data/工况/'+wenjian+'/bw.csv')
    df_name = pd.read_csv('static/data/biwen_name.csv', encoding='gb2312')
    column_headers = df.columns.values
    column_headers = np.array(column_headers)

    # dff1只为读取对应的时间
    dff1 = pd.read_csv("./static/data/工况/"+ wenjian + '/bw.csv')
    dataa = dff1.iloc[:, 0].values
    time = dataa.tolist()

    timebb = time[0]
    timeo = time[-1]

    choose_id = request.form.get('choose_id')
    print("选取框时的ID:",choose_id)
    indexKaung = request.form.get("index")
    indexKaung = json.loads(indexKaung)

    print("indexkuang2",indexKaung)

    # 获取选矿的值 区间为(5-60) 步长为5
    valuerange = request.form.get("valuerange")
    print("valuerange",valuerange)
    valuerange = int(valuerange)

    # 例如: 10s内是两行数据
    timezu = int(valuerange/5)
    print("timezu",timezu)

    # erced 最后一个值表示 当前ID的全部数据的平均值
    erced,ma,mi = cedianwz(choose_id,wenjian)

    # 获取选取ID的值
    xuanquID = []
    for i in range(len(indexKaung)):
        print("选取的ID:",erced[indexKaung[i]][2])
        xuanquID.append(erced[indexKaung[i]][2])
    print(xuanquID)

    # 获取index 对应的 ID 坐标
    mId = []
    for j in range(len(xuanquID)):
        for i in range(len(column_headers)):
            if (xuanquID[j] in column_headers[i]):
                mId.append(i)

    print("mId",mId)
    data = df.iloc[:, mId].values
    # print(data)

    # 获取每一行的均值
    # print("每一行均值:")
    datajun = np.mean(data, axis=1)
    # print(datajun)
    datjunlist = datajun.tolist()
    # print(datjunlist)

    # 定义分割后的均值数据
    datajunfen = []

    # print("length",len(datjunlist))
    b = [datjunlist[i:i+timezu] for i in range(0,len(datjunlist),timezu)]
    # 刪除列表最后一个不规则数据源
    b = b[0:-1]
    # print("分割后数据:",b)
    for i in range(len(b)):
        sumjun = 0
        junfen = 0

        for j in range(timezu):
            sumjun = sumjun + b[i][j]
            junfen = float(sumjun / timezu)

        datajunfen.append(junfen)

    print("datajunfen",datajunfen)
    time = df.iloc[:, 0].values
    time = time.tolist()
    timeb = [time[i:i + timezu] for i in range(0, len(time), timezu)]
    # print("timeb",timeb)
    timefen = []
    for i in range(len(timeb)):
        timefen.append(timeb[i][0])

    print("timefen",timefen)
    timenew = []
    for i in range(len(timefen)):
        a = timefen[i]
        timenew.append(a[10:])

    return jsonify({
        "datajunfen" : datajunfen,
        "time":timenew,
        "timeb":timebb,
        "timeo":timeo
    })


# 定义 二维图点击事件 触发该事件 展示单个壁温测点的数据
@app.route('/xuanqu3',methods=['GET','POST'])
def xuanqu3():
    # 获取前端传过来的值
    wenjian = request.form.get('wenjian')
    print(wenjian)
    df = pd.read_csv('static/data/工况/'+wenjian+'/bw.csv')
    df_name = pd.read_csv('static/data/biwen_name.csv', encoding='gb2312')
    column_headers = df.columns.values
    column_headers = np.array(column_headers)

    # dff1只为读取对应的时间
    dff1 = pd.read_csv("./static/data/工况/"+ wenjian + '/bw.csv')
    dataa = dff1.iloc[:, 0].values
    time = dataa.tolist()
    timeb = time[0]
    timeo = time[-1]
    print("timeb",timeb)
    print("timeo",timeo)

    # 存储的是截取后的时-分数据
    timenew = []
    for i in range(len(time)):
        a = time[i]
        # print("---",a)
        timenew.append(a[10:])

    # print("timenew",timenew)

    choose_id = request.form.get('choose_id')
    print("选取框时的ID:",choose_id)
    indexkuang3 = int(request.form.get("index"))

    print("indexkuang3",indexkuang3)

    # erced 最后一个值表示 当前ID的全部数据的平均值
    erced,ma,mi = cedianwz(choose_id,wenjian)

    xuanquid = erced[indexkuang3][-1]
    print("xuanquid",xuanquid)

    # 获取index 对应的 ID 坐标
    mId = []

    for i in range(len(column_headers)):
        if (xuanquid in column_headers[i]):
            mId.append(i)

    # print("mId",mId)
    a = mId[0]
    # print("a",a+1)
    data = df.iloc[:, a].values
    # print("clcik one data",data)
    # print(data.tolist())

    return jsonify({
        'data' : data.tolist(),
        "time" : timenew,  # 前2000行时间
        "mingzi":xuanquid,
        "timeb":timeb,
        "timeo":timeo
    })


# 点击单个壁温的 + 时间拖选框事件
@app.route('/xuanqu4',methods=['GET','POST'])
def xuanqu4():
    # df = pd.read_csv("static/data/biwen_data.csv")
    wenjian=request.form.get('wenjian')
    print(wenjian)
    df = pd.read_csv('static/data/工况/'+wenjian+'/bw.csv')
    df_name = pd.read_csv('static/data/biwen_name.csv', encoding='gb2312')
    column_headers = df.columns.values
    column_headers = np.array(column_headers)

    # dff1只为读取对应的时间
    dff1 = pd.read_csv("./static/data/工况/"+ wenjian + '/bw.csv')
    dataa = dff1.iloc[:, 0].values
    time = dataa.tolist()

    timebb = time[0]
    timeo = time[-1]

    choose_id = request.form.get('choose_id')
    print("选取框时的ID:",choose_id)
    indexKaung = request.form.get("index")
    indexKaung = json.loads(indexKaung)

    print("indexkuang",indexKaung)

    # 获取选矿的值 区间为(5-60) 步长为5
    valuerange = request.form.get("valuerange")
    print("valuerange",valuerange)
    valuerange = int(valuerange)

    # 例如: 10s内是两行数据
    timezu = int(valuerange/5)
    print("timezu",timezu)

    # erced 最后一个值表示 当前ID的全部数据的平均值
    erced,ma,mi = cedianwz(choose_id,wenjian)
    # print("erced",erced)

    # 获取选取ID的值
    xuanquID = []
    print("选取的ID:",erced[indexKaung][2])
    xuanquID.append(erced[indexKaung][2])
    print(xuanquID)

    # 获取index 对应的 ID 坐标
    mId = []
    for j in range(len(xuanquID)):
        for i in range(len(column_headers)):
            if (xuanquID[j] in column_headers[i]):
                mId.append(i)

    print("mId",mId)
    data = df.iloc[:, mId].values
    # print(data)

    # 获取每一行的均值
    # print("每一行均值:")
    datajun = np.mean(data, axis=1)
    # print(datajun)
    datjunlist = datajun.tolist()
    # print(datjunlist)

    # 定义分割后的均值数据
    datajunfen = []

    # print("length",len(datjunlist))
    b = [datjunlist[i:i+timezu] for i in range(0,len(datjunlist),timezu)]
    # 刪除列表最后一个不规则数据源
    b = b[0:-1]
    # print("分割后数据:",b)
    for i in range(len(b)):
        sumjun = 0
        junfen = 0

        for j in range(timezu):
            sumjun = sumjun + b[i][j]
            junfen = float(sumjun / timezu)

        datajunfen.append(junfen)

    print("datajunfen",datajunfen)
    time = df.iloc[:, 0].values
    time = time.tolist()
    timeb = [time[i:i + timezu] for i in range(0, len(time), timezu)]
    # print("timeb",timeb)
    timefen = []
    for i in range(len(timeb)):
        timefen.append(timeb[i][0])

    print("timefen",timefen)
    timenew = []
    for i in range(len(timefen)):
        a = timefen[i]
        timenew.append(a[10:])

    return jsonify({
        "datajunfen" : datajunfen,
        "time":timenew,
        "timeb":timebb,
        "timeo":timeo
    }
    )

@app.route('/FileVis',methods=['GET','POST'])
def FileVis():
    filelist = request.form.get("filelist")
    filelist = json.loads(filelist)

    # 表示所选取的功率文件夹
    wenjian = request.form.get("wenjian")
    print("know wenjian",wenjian)
    print("lalala")
    print("filelist",filelist)
    allname = []
    datalist = []

    # dff1只为读取对应的时间
    dff1 = pd.read_csv("./static/data/工况/"+ wenjian + "/" + filelist[0] + ".csv")
    dataa = dff1.iloc[:, 0].values
    time = dataa.tolist()

    timenew = []

    for i in range(len(time)):
        a = time[i]
        # print("---",a)
        timenew.append(a[10:])

    # print("timenew",timenew)

    # 读取整体壁温数据
    df = pd.read_csv("./static/data/工况/"+ wenjian + "/" + "bw.csv")
    column_headers = df.columns.values
    column_headers = column_headers.tolist()

    # 读取功率数据
    df2 = pd.read_csv("./static/data/工况/" + wenjian + "/" + "qita.csv")
    datagong = df2.iloc[:,213].values
    datagong = datagong.tolist()
    for i in range(len(datagong)):
        if math.isnan(datagong[i]):
            datagong[i] = datagong[i - 1]

    # 遍历文件ID
    for i in range(len(filelist)):
        num = re.findall(r"\d+", filelist[i])
        num = int(num[0]) + 1
        print("filelist[i]",filelist[i][:-5])
        for j in range(1,num):
            name = filelist[i][:-5] + "FT" + str(j)
            allname.append(name)
            allname.append("AVG" + str(j))

    # allname 是为了前端生成legend图例
    print("allname",allname)
    for i in range(len(allname)):
        if "高温过热器" in allname[i]:
            allname[i] = allname[i].replace('高温过热器','HTS')
        elif "低温过热器上部屏" in allname[i]:
            allname[i] = allname[i].replace('低温过热器上部屏','LTS(U)')
        elif "低温过热器下部屏" in allname[i]:
            allname[i] = allname[i].replace('低温过热器下部屏', 'LTS(L)')
        elif "后墙隔墙" in allname[i]:
            allname[i] = allname[i].replace('后墙隔墙','RW')

    allname.append('MW')
    print(allname)


    # 计算并遍历融合前的平均数据 (根据聚类的类别---获取数据---求平均)
    for i in range(len(filelist)):
        df1 = pd.read_csv("./static/data/工况/"+ wenjian + "/" + filelist[i] + ".csv")
        num = re.findall(r"\d+", filelist[i])
        num = int(num[0]) + 1
        print("num",num)
        english = ''.join(re.findall(r"[a-zA-Z]", filelist[i]))
        print("english",english)
        pattern = re.compile('(.[\u4E00-\u9FA5]+)')
        content = pattern.findall(filelist[i])
        content = content[0]
        dataid = []
        columns = []
        cluster_num = num-1

        # 读取已经聚类后保存的txt文件 获取所保存的聚类类别
        path_out = pathbase + "/static/data/工况/" + wenjian + '/' + filelist[i] + '.txt'
        b = open(path_out, "r", encoding='UTF-8')
        out = b.read()
        out = json.loads(out)
        print("----erced ----")
        erced = out['erced']
        print(erced)
        labels1 = []
        for i in range(len(erced)):
            labels1.append(erced[i][-1])

        labels1 = np.array(labels1)
        print("labels1", labels1)
        print(labels1.shape)

        if "水冷左垂" in content:
            content = '左侧墙垂直'
        elif "水冷右垂" in content:
            content = '右侧墙垂直'
        elif "水冷后垂" in content:
            content = '后墙垂直'
        elif "水冷前垂" in content:
            content = '前墙垂直'
        elif "水冷左螺" in content:
            content = '左侧墙螺旋'
        elif "水冷右螺" in content:
            content = '右侧墙螺旋'
        elif "水冷前螺" in content:
            content = '前墙螺旋'
        elif "水冷后螺" in content:
            content = '后墙螺旋'
        elif "低温过热器上部屏" in content:
            content = '低温过热器上部屏出口段后墙横向'
        elif "低温过热器上部屏" in content:
            content = '低温过热器下部屏出口段后墙横向'
        else:
            content = content

        print("content",content)

        if "SC" in english:
            cluster_name = 'SpectralCluster'
        elif "KMedoids" in english:
            cluster_name = 'KMedoids'
        elif "AC" in english:
            cluster_name = 'AgglomerativeCluster'
        elif "ACDTW" in english:
            cluster_name = "AgglomerativeCluster_DTW"
        else:
            cluster_name = english
        print("cluster_name",cluster_name)

        for k in range(len(column_headers)):
            if content in column_headers[k]:
                columns.append(column_headers[k])
                dataid.append(k)
        print("dataid",dataid)
        print("columns",columns)

        # 读取该文件的壁温数据
        data = df.iloc[:, dataid].values

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if math.isnan(data[i, j]):
                    data[i, j] = data[i - 1, j]

        # 读取计算好的DTW距离
        df22 = pd.read_csv('static/data/juli3.csv')
        data2 = df22.iloc[:, :].values
        data2 = data2[:, dataid]
        data2 = data2[dataid, :]

        # 获取聚类后的类别
        labels = cluster2(data.T,data2,cluster_name,cluster_num)
        print("labels",labels)
        print("labels_shape",labels.shape)

        for i in range(0,cluster_num):
            indexxx = np.argwhere(labels1==i).reshape(-1,)
            j = i + 1
            print("indexxx",indexxx)
            # 存储聚类后的数据
            data_ju = data[:,indexxx]
            print("indexxx_data",data_ju)
            datajun = np.mean(data_ju,axis=1)
            print("datajun",datajun)
            datajun = datajun.tolist()
            # print("datajun.shape",datajun.shape)

            # 读取该数据对应的的融合数据
            data1 = df1.iloc[:,j].values
            print("data1.shape",data1.shape)
            print("data1",data1)
            data1 = data1.tolist()
            print("new data1-list",data1)

            datalist.extend([data1])
            datalist.extend([datajun])

    datalist.extend([datagong])
    print("datalist", datalist)
    print("datalist length",len(datalist))

    return jsonify({
        "shuju" : datalist,
        "time" : timenew,
        "allname" : allname
    })



@app.route('/BatchMse',methods=['GET','POST'])
def BatchMse():
    filelist = request.form.get("filelist")
    filelist = json.loads(filelist)
    print("lalala")
    print("filelist", filelist)

    wenjian = request.form.get("wenjian")
    print("wenjian",wenjian)

    df = pd.read_csv("./static/data/工况/"+ wenjian + "/" + "bw.csv")
    column_headers = df.columns.values
    column_headers = column_headers.tolist()

    datamselist = []
    datarmselist = []
    datamaelist = []
    datamapelist = []

    datajunrmse = []
    datajunmae = []
    datajunmape = []


    allname = []
    # 遍历文件ID
    for i in range(len(filelist)):
        num = re.findall(r"\d+", filelist[i])
        num = int(num[0]) + 1
        for j in range(1,num):
            name = filelist[i][:-5] + "FT" + str(j)
            allname.append(name)

    print("allname",allname)
    for i in range(len(allname)):
        if "高温过热器" in allname[i]:
            allname[i] = allname[i].replace('高温过热器','HTS')
        elif "低温过热器上部屏" in allname[i]:
            allname[i] = allname[i].replace('低温过热器上部屏','LTS(U)')
        elif "低温过热器下部屏" in allname[i]:
            allname[i] = allname[i].replace('低温过热器下部屏', 'LTS(L)')
        elif "后墙隔墙" in allname[i]:
            allname[i] = allname[i].replace('后墙隔墙','RW')
    print("allname1",allname)

    allname.append("Sum")
    allname.append("Arithmetic Mean")


    # 遍历数据
    for i in range(len(filelist)):
        df1 = pd.read_csv("./static/data/工况/"+ wenjian + "/" + filelist[i] + ".csv")
        num = re.findall(r"\d+", filelist[i])
        num = int(num[0]) + 1
        print("num",num)
        english = ''.join(re.findall(r"[a-zA-Z]", filelist[i]))
        print("english",english)
        pattern = re.compile('(.[\u4E00-\u9FA5]+)')
        content = pattern.findall(filelist[i])
        content = content[0]
        dataid = []
        columns = []
        cluster_num = num-1

        # print("old content",content)

        # 读取已经保存后的聚类类别数据
        path_out = pathbase + "/static/data/工况/" + wenjian + '/' + filelist[i] + '.txt'
        b = open(path_out, "r", encoding='UTF-8')
        out = b.read()
        out = json.loads(out)
        print("----erced ----")
        erced = out['erced']
        print(erced)
        labels1 = []
        for i in range(len(erced)):
            labels1.append(erced[i][-1])

        labels1 = np.array(labels1)
        print("labels1", labels1)
        print(labels1.shape)

        if "水冷左垂" in content:
            content = '左侧墙垂直'
        elif "水冷右垂" in content:
            content = '右侧墙垂直'
        elif "水冷后垂" in content:
            content = '后墙垂直'
        elif "水冷前垂" in content:
            content = '前墙垂直'
        elif "水冷左螺" in content:
            content = '左侧墙螺旋'
        elif "水冷右螺" in content:
            content = '右侧墙螺旋'
        elif "水冷前螺" in content:
            content = '前墙螺旋'
        elif "水冷后螺" in content:
            content = '后墙螺旋'
        elif "低温过热器上部屏" in content:
            content = '低温过热器上部屏出口段后墙横向'
        elif "低温过热器下部屏" in content:
            content = '低温过热器下部屏出口段后墙横向'
        else:
            content = content

        print("content",content)

        if "SC" in english:
            cluster_name = 'SpectralCluster'
        elif "KMedoids" in english:
            cluster_name = 'KMedoids'
        elif "AC" in english:
            cluster_name = 'AgglomerativeCluster'
        elif "ACDTW" in english:
            cluster_name = "AgglomerativeCluster_DTW"
        else:
            cluster_name = english
        print("cluster_name",cluster_name)

        for k in range(len(column_headers)):
            if content in column_headers[k]:
                columns.append(column_headers[k])
                dataid.append(k)
        print("dataid",dataid)
        print("columns",columns)

        # 读取该文件的壁温数据
        data = df.iloc[:, dataid].values

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if math.isnan(data[i, j]):
                    data[i, j] = data[i - 1, j]

        # 计算整体数据集的平均值
        alldatajun = np.mean(data, axis=1)
        print(alldatajun)
        print("alldatajun.shape",alldatajun.shape)

        # 读取计算好的DTW距离
        df2 = pd.read_csv('static/data/juli3.csv')
        data2 = df2.iloc[:, :].values
        data2 = data2[:, dataid]
        data2 = data2[dataid, :]

        # 重新计算并获取聚类后的类别
        labels = cluster2(data.T,data2,cluster_name,cluster_num)
        print("labels",labels)
        print("labels_shape",labels.shape)

        for i in range(0,cluster_num):
            indexxx = np.argwhere(labels1==i).reshape(-1,)
            j = i + 1
            print("indexxx",indexxx)
            # 存储聚类后的数据
            data_ju = data[:,indexxx]
            print("indexxx_data",data_ju)
            datajun = np.mean(data_ju,axis=1)
            print("datajun",datajun)
            print("datajun.shape",datajun.shape)

            # 读取该数据对应的的融合数据
            data1 = df1.iloc[:,j].values
            print("data1.shape",data1.shape)
            print("data1",data1)
            data1 = data1.tolist()
            # print("data1-list",data1)

            # 计算整体平均值和聚类均值之间的误差
            datarmse1 = rmse(alldatajun,datajun)
            datajunrmse.append(datarmse1)

            datamae1 = mae(alldatajun,datajun)
            datajunmae.append(datamae1)

            datamape1 = mape(alldatajun,datajun)
            datajunmape.append(datamape1)


            # 分别计算不同的误差评价系数
            datamse = mse(datajun,data1)
            datamselist.extend([datamse])

            datarmse = rmse(datajun,data1)
            datarmselist.extend([datarmse])

            datamae = mae(datajun,data1)
            datamaelist.extend([datamae])

            datamape = mape(datajun,data1)
            datamapelist.extend([datamape])


        sum1 = 0
        sum2 = 0
        sum3 = 0
        for i in range(len(datajunrmse)):
            print("---datajunrmse---")
            print(datajunrmse[i])

            sum1 = sum1 + datajunrmse[i]
            sum2 = sum2 + datajunmae[i]
            sum3 = sum3 + datajunmape[i]

        datajunrmse1 = round(sum1/len(datajunrmse),2)
        datajunmae1 = round(sum2/len(datajunrmse),2)
        datajunmape1 = round(sum3/len(datajunrmse), 2)

        rongrmse = 0
        rongmae = 0
        rongmape = 0

        # print("asdasdasd",len(datarmselist))
        # 添加融合误差数据的平均误差数据
        for i in range(len(datarmselist)):
            rongrmse = rongrmse + datarmselist[i]
            rongmae = rongmae + datamaelist[i]
            rongmape = rongmape + datamapelist[i]


        print("rongrmse",rongrmse)
        print("rongmae", rongmae)
        print("rongmape", rongmape)

        rongjunrmse = round(rongrmse/cluster_num,2)
        rongjunmae = round(rongmae/cluster_num,2)
        rongjunmape = round(rongmape/cluster_num,2)

        print("rongjunrmse",rongjunrmse)
        print("rongjunmae", rongjunmae)
        print("rongjunmape", rongjunmape)

        # 先添加融合的误差平均数据
        datarmselist.append(rongjunrmse)
        datamaelist.append(rongjunmae)
        datamapelist.append(rongjunmape)

        #再添加整体的平均数据
        datarmselist.append(datajunrmse1)
        datamaelist.append(datajunmae1)
        datamapelist.append(datajunmape1)

        # print("datamselist" , datamselist)
        # print("datarmselist", datarmselist)
        # print("datamaelist", datamaelist)
        # print("datamapelist", datamapelist)

    return jsonify({
        "datamselist" : datamselist,
        "datarmselist": datarmselist,
        "datamaelist" : datamaelist,
        "datamapelist": datamapelist,
        "allname" : allname
    })



@app.route('/yinzi',methods=['GET','POST'])
def yinzi():
    choose_id=request.form.get('choose_id')
    if choose_id=='高温过热器出口段横向':
        df = pd.read_csv('static/data/高温过热器因子得分.csv')
    elif choose_id=='低温过热器上部屏出口段后墙横向':
        df = pd.read_csv('static/data/低过上后因子得分.csv')
    elif choose_id=='低温过热器下部屏出口段后墙横向':
        df = pd.read_csv('static/data/低过下后因子得分.csv')
    elif choose_id=='后墙隔墙出口段':
        df = pd.read_csv('static/data/后墙隔墙因子得分.csv',encoding='gb18030')
    # df = pd.read_csv('static/data/高温过热器因子得分.csv')
    data = df.iloc[:, :].values
    # print(data)
    data1=data[:,0]
    data1=data1.reshape((-1,1))
    data2=data[:,1:]
    data=np.hstack((data2,data1))
    # data=np.round(data,4)
    # choose_id=request.form.get('choose_id')
    erced=cedianwz(choose_id)
    return jsonify({
        "data1": data.tolist(),
        "erced":erced
    })

@app.route('/CCA',methods=['GET','POST'])
def CCA():
    input_list=request.form.get('input')
    input_list=json.loads(input_list)
    output_list=request.form.get('output')
    output_list=json.loads(output_list)
    index_list=request.form.get('index')
    index_list=json.loads(index_list)
    # print(output_list)
    # print(input_list)
    # print(index_list)
    # print(cca_name[index_list])
    # print(index_list[0:len(input_list)])
    # print(index_list[len(input_list):])
    # print(cca_data)
    # print(cca_data.shape)
    X=pd.DataFrame(cca_data[:,index_list[0:len(input_list)]],columns=input_list)
    Y=pd.DataFrame(cca_data[:,index_list[len(input_list):]],columns=output_list)
    # print(X.shape,Y.shape)
    a,b,dcoef1,jieshi,jianyan,corr_xv,corr_yu,corr_xu,corr_yv=cca_as(X, Y)
    jieshi=np.round(jieshi,3)
    # print(jianyan)
    # print(dcoef1)
    dcoef1=dcoef1.reshape((-1,1))
    jy=np.hstack((dcoef1,jianyan))
    # print(jy)
    # print(xishu)
    return jsonify({
        "jieshi": jieshi.tolist(),
        "jy":jy.tolist(),
        "a":a.T.tolist(),
        "b":b.T.tolist()
    })

def cca_as(x, y):
    # import numpy as np
    # import pandas as pd
    # from scipy import stats  # 计算p值
    '''
    输入：
        x：X组的矩阵/DataFrame，行是样本，列是变量
        y：Y组的矩阵/DataFrame，行是样本，列是变量
        注：列名不能相同，否则会报错
    输出：
        输出一个excel('典型相关分析输出结果.xlsx')：
        X组的典型相关变量_系数
        Y组的典型相关变量_系数
        xu的相关系数; xv的相关系数; yv的相关系数; yu的相关系数
        解释百分比
        显著性检验
    '''

    r = x.join(y).corr()  # 计算x和y合并后的相关系数
    n1 = x.shape[1]  # X组变量个数
    n2 = y.shape[1]  # Y组变量个数
    r = np.array(r)  # X和Y的相关系数转为矩阵
    num = min(n1, n2)
    print("n1,n2",n1,n2,num)
    s1 = r[:n1, :n1]  # 提出X与X的相关系数
    s2 = r[n1:, n1:]  # 提出Y与Y的相关系数
    s12 = r[:n1, n1:]  # 提出X与Y的相关系数
    s21 = s12.T  # Y与X的相关系数
    # print("x",x)
    # print("y",y)
    # print("r",r)
    # print("nl",np.linalg.inv(s2))
    # print("nl2",s1)
    m1 = np.linalg.inv(s1).dot(s12).dot(np.linalg.inv(s2)).dot(s21)  # 计算矩阵M1
    m2 = np.linalg.inv(s2).dot(s21).dot(np.linalg.inv(s1)).dot(s12)  # 计算矩阵M2
    print("m1",m1.shape,m2.shape)
    writer = pd.ExcelWriter('典型相关分析ces.xlsx')  # 打开excel文件

    # ---------------------------计算X组的典型相关向量和典型相关系数------------------------
    [val1, vec1] = np.linalg.eig(m1)  # M1的特征值vec1和特征向量val1
    val1 = val1.real
    vec1 = vec1.real
    # print("val1",val1,vec1)
    for i in range(n1):
        # 特征向量归一化，满足a's1a=1
        vec1[:, i] = vec1[:, i] / np.sqrt(vec1[:, i].T.dot(s1).dot(vec1[:, i]))
        # 特征向量乘+-1，保证所有分量为正
        vec1[:, i] = vec1[:, i] / np.sign(np.sum(vec1[:, i]))

    val1 = np.sqrt(val1[val1 >= 0])  # 计算特征值的平方根
    print("know val1",val1)
    val1, ind1 = np.sort(val1, axis=0)[::-1], np.argsort(-val1, axis=0)  # 按照从大到小排列
    # print("val2",val1,ind1,vec1)
    # print(ind1.flatten())
    a = vec1[:, ind1.flatten()[:num]]  # 取出x组的系数阵a
    # print(a)
    dcoef1 = val1.flatten()[:num]  # 提出典型相关系数
    print("know dcoef1",dcoef1)
    save1 = pd.DataFrame(a, columns=['u%s' % i for i in range(1, num + 1)],
                         index=['x%s' % i for i in range(1, n1 + 1)])  # 转为DataFrame
    # print('X组的系数阵/典型相关变量：');
    # display(save1)
    save1.loc['典型相关系数'] = dcoef1
    save1.to_excel(writer, 'X组的典型相关变量_系数')  # 系数矩阵

    # ----------------------------计算Y组的典型相关向量和典型相关系数-------------------------
    [val2, vec2] = np.linalg.eig(m2)  # m2的特征值vec2和特征向量val2
    val2 = val2.real
    vec2 = vec2.real  # 这里得到的结果是复数，所以要提取为实数
    print('ceshi')
    print(val1.shape,val2.shape,val2)
    for i in range(n2):
        # 特征向量归一化，满足b's2b=save2
        vec2[:, i] = vec2[:, i] / np.sqrt(vec2[:, i].T.dot(s2).dot(vec2[:, i]))
        # 特征向量乘+-1，保证所有分量为正
        vec2[:, i] = vec2[:, i] / np.sign(np.sum(vec2[:, i]))

    val2 = np.sqrt(val2[val2 >= 0])  # 计算特征值的平方根
    print(val2.shape)
    val2, ind2 = np.sort(val2)[::-1], val2.argsort()[::-1]  # 选出大于0的，并从大到小排序得到索引
    # print("val3",val2,ind2,vec2.shape)
    b = vec2[:, ind2[:num]]  # 取出x组的系数阵a
    dcoef2 = val2[:num]  # 提出典型相关系数
    # print("ceshi")
    # print(vec2.shape)
    # print(val2.shape,val2)
    # print(['v%s' % i for i in range(1, num + 1)])
    # print(['y%s' % i for i in range(1, n2 + 1)])
    save2 = pd.DataFrame(b, columns=['v%s' % i for i in range(1, num + 1)],
                         index=['y%s' % i for i in range(1, n2 + 1)])  # 转为DataFrame
    # print('Y组的系数阵/典型相关变量：');
    # display(save2)
    # print('典型相关系数：');
    # print(dcoef2)
    save2.loc['典型相关系数'] = dcoef2
    save2.to_excel(writer, 'Y组的典型相关变量_系数')  # 系数矩阵

    # ------------------------------计算原始变量和典型变量的相关系数---------------------------
    rxu = s1.dot(a)  # xu的相关系数
    # jisuan_x=x.values
    # print("rxu",rxu)
    corr_xu=[]
    for i in range(a.shape[1]):
        jisuan_x=x.values
        jisuan_x -= np.mean(x.values,axis=0)
        jisuan_x /= np.std(x.values,axis=0)
        # print(jisuan_x.shape,a.shape)
        jisuan_x=jisuan_x.dot(a[:,i])
        # print(a[:,i])
        # print("jisuan_x",jisuan_x)
        jisuan_x=jisuan_x.reshape(-1,1)
        # print(jisuan_x.shape)
        jisuan_x2=y.values
        jisuan_x2 -= np.mean(y.values,axis=0)
        jisuan_x2 /= np.std(y.values,axis=0)
        jisuan_x2=jisuan_x2.dot(b[:,i])
        jisuan_x2=jisuan_x2.reshape(-1,1)
        jisuan_x3=np.hstack((jisuan_x,jisuan_x2))
        jisuan_corr2=np.corrcoef(jisuan_x3.T)
        print(jisuan_corr2)

        jisuan_x=np.hstack((jisuan_x,x.values))
        # print(jisuan_x)
        # print(jisuan_x.shape)
        jisuan_corr=np.corrcoef(jisuan_x.T)
        # print("jisuan_corr",jisuan_corr)
        # print(jisuan_corr.shape)
        jisuan_corr=jisuan_corr[1:,0]
        corr_xu.extend([list(jisuan_corr)])
        # print(corr_xu)
    corr_xu=np.array(corr_xu)
    # corr_xu=corr_xu.T

    corr_xv=[]
    for i in range(b.shape[1]):
        jisuan_x=y.values
        jisuan_x=jisuan_x.dot(b[:,i])
        # print("jisuan_x",jisuan_x)
        jisuan_x=jisuan_x.reshape(-1,1)
        # print(jisuan_x.shape)
        jisuan_x=np.hstack((jisuan_x,x.values))
        # print(jisuan_x)
        # print(jisuan_x.shape)
        jisuan_corr=np.corrcoef(jisuan_x.T)
        # print(jisuan_corr)
        # print(jisuan_corr.shape)
        jisuan_corr=jisuan_corr[1:,0]
        corr_xv.extend([list(jisuan_corr)])
    corr_xv=np.array(corr_xv)
    # corr_xv=corr_xv.T

    corr_yu=[]
    for i in range(a.shape[1]):
        jisuan_x=x.values
        jisuan_x=jisuan_x.dot(a[:,i])
        # print("jisuan_x",jisuan_x)
        jisuan_x=jisuan_x.reshape(-1,1)
        # print(jisuan_x.shape)
        jisuan_x=np.hstack((jisuan_x,y.values))
        # print(jisuan_x)
        # print(jisuan_x.shape)
        jisuan_corr=np.corrcoef(jisuan_x.T)
        # print(jisuan_corr)
        # print(jisuan_corr.shape)
        jisuan_corr=jisuan_corr[1:,0]
        corr_yu.extend([list(jisuan_corr)])
    corr_yu=np.array(corr_yu)
    # corr_yu=corr_yu.T

    corr_yv=[]
    for i in range(b.shape[1]):
        jisuan_x=y.values
        jisuan_x=jisuan_x.dot(b[:,i])
        # print("jisuan_x",jisuan_x)
        jisuan_x=jisuan_x.reshape(-1,1)
        # print(jisuan_x.shape)
        jisuan_x=np.hstack((jisuan_x,y.values))
        # print(jisuan_x)
        # print(jisuan_x.shape)
        jisuan_corr=np.corrcoef(jisuan_x.T)
        # print(jisuan_corr)
        # print(jisuan_corr.shape)
        jisuan_corr=jisuan_corr[1:,0]
        corr_yv.extend([list(jisuan_corr)])
    corr_yv=np.array(corr_yv)
    # corr_yv=corr_yv.T

    # print(s1.shape)
    # print(s1)
    # print(a)
    # print(a.shape)
    ryv = s2.dot(b)  # yv的相关系数
    rxv = s12.dot(b)  # xv的相关系数
    ryu = s21.dot(a)  # yu的相关系数
    x_index = ['x%s' % i for i in range(1, n1 + 1)]
    y_index = ['y%s' % i for i in range(1, n2 + 1)]
    u_col = ['u%s' % i for i in range(1, num + 1)]
    v_col = ['v%s' % i for i in range(1, num + 1)]
    pd.DataFrame(rxu, columns=u_col, index=x_index).to_excel(writer, 'xu的相关系数')
    pd.DataFrame(rxv, columns=v_col, index=x_index).to_excel(writer, 'xv的相关系数')
    pd.DataFrame(ryv, columns=v_col, index=y_index).to_excel(writer, 'yv的相关系数')
    pd.DataFrame(ryu, columns=u_col, index=y_index).to_excel(writer, 'yu的相关系数')

    # -------------------------------计算解释的方差比例----------------------------------------
    mu = np.sum(rxu ** 2) / n1;
    # print('*x组原始变量被u_i解释的方差比例：', format(mu, '.2%'))
    mv = np.sum(rxv ** 2) / n1;
    # print('x组原始变量被v_i解释的方差比例：', format(mv, '.2%'))
    nv = np.sum(ryv ** 2) / n2;
    # print('*y组原始变量被v_i解释的方差比例：', format(nv, '.2%'))
    nu = np.sum(ryu ** 2) / n2;
    # print('y组原始变量被u_i解释的方差比例：', format(nu, '.2%'))
    pd.DataFrame([mu, mv, nv, nu], index=['u解释x', 'v解释x', 'v解释y', 'u解释y']).to_excel(writer, '解释百分比')

    # -------------------------------典型相关系数的卡方检验------------------------------------
    # 典型相关系数就是特征值，逐一对典型相关系数做检验
    jianyan = pd.DataFrame(columns=['自由度', '卡方计算值', '卡方临界值(0.05显著水平)', 'P值'], index=range(1, num + 1))
    for j in range(num):  # 遍历num个典型相关系数
        k = j + 1;
        n = min(90,len(x))  # 第k个λ，n是样本量
        prod_lamuda = np.prod([1 - i ** 2 for i in dcoef1[j:]])  # 拉姆达连乘
        df = (n1 - k + 1) * (n2 - k + 1)  # 自由度
        Q = -(n - k - 0.5 * (n1 + n2 + 1)) * np.log(prod_lamuda)  # 统计量
        p_value = 1 - stats.chi2.cdf(Q, df)  # p值
        # print("nn",n,Q,prod_lamuda,np.log(prod_lamuda))
        jianyan.loc[k, '自由度'] = df
        jianyan.loc[k, '卡方计算值'] = Q
        jianyan.loc[k, '卡方临界值(0.05显著水平)'] = stats.chi2.ppf(0.95, df)
        jianyan.loc[k, 'P值'] = p_value
    # display(jianyan)
    jianyan.to_excel(writer, '显著性检验')
    writer.save()  # 保存excel文件
    # xishu=np.vstack((a,b))
    jieshi=np.zeros((4,rxu.shape[1]))
    # print("rxu",rxu.shape,ryu.shape)
    for i in range(rxu.shape[1]):
        count1=0
        count2=0
        # print(corr_xu[i])
        for j in range(rxu.shape[0]):
            # print(i,count1,rxu[j,i],rxv[j,i])
            count1=count1+rxu[j,i]**2
            count2=count2+rxv[j,i]**2
        # print("count2",count1,count2)
        jieshi[0,i]=count1/corr_xu.shape[0]
        jieshi[1,i]=count2/corr_xu.shape[0]
        # print(jieshi[0,i],jieshi[1,i])
        # print(jieshi)
    # print("jieshi",jieshi)
    # print(corr_yu.shape)
    # print(corr_yv.shape)
    for i in range(ryu.shape[1]):
        count1=0
        count2=0
        # print(corr_yu[i])
        for j in range(ryu.shape[0]):
            # print(i,count1,ryu[j,i],ryv[j,i])
            count1=count1+ryu[j,i]**2
            count2=count2+ryv[j,i]**2
        jieshi[2,i]=count1/corr_yu.shape[0]
        jieshi[3,i]=count2/corr_yu.shape[0]
    # for i in range(ryu.shape[0]):
    #     for j in range(ryu.shape[1]):
    #         jieshi[2,i]=jieshi[2,i]+ryu[i,j]**2
    #         jieshi[3,i]=jieshi[3,i]+ryv[i,j]**2
    #     jieshi[2,i]=jieshi[2,i]/ryu.shape[0]
    #     jieshi[3,i]=jieshi[3,i]/ryu.shape[0]
    # print(jieshi)
    #     # print(jianyan.values)
    # print(xishu)
    return a,b,dcoef1,jieshi,jianyan.values[:,1:3],rxv,ryu,rxu,ryv

def cedianwz(choose_id,wenjian):
    dff = pd.read_csv('static/data/工况/'+wenjian+'/bw.csv')
    column_headers=np.array(dff.columns.values[1:])
    dataa = dff.iloc[:, 1:].values
    # print("dataa",dataa)
    for i in range(dataa.shape[0]):
        for j in range(1,dataa.shape[1]):
            if math.isnan(dataa[i,j]):
                dataa[i,j]=dataa[i-1,j]
    # df = pd.read_csv('static/data/biwen_name.csv',encoding='gb2312')
    # column_headers = df.columns.values
    # column_headers=np.array(column_headers)
    # print("column_headers",column_headers)

    # 读取水冷壁温数据
    # dff1 = pd.read_csv("static/data/ShuiWen2.csv")
    # dff1 = pd.read_csv('static/data/工况/'+wenjian+'/qita.csv')
    # dataa1 = dff1.iloc[:].values
    # for i in range(dataa1.shape[0]):
    #     for j in range(1,dataa1.shape[1]):
    #         if math.isnan(dataa1[i,j]):
    #             dataa1[i,j]=dataa1[i-1,j]

    # 读取水冷壁温的列名
    # df1 = pd.read_csv("static/data/ShuiWen2.csv")
    # column_headers1 = df1.columns.values
    # column_headers1 = np.array(column_headers1)
    # print("column_headers1",column_headers1)

    # 定义水冷壁温 ID

    # 水垂直左侧壁温
    scz_id = []
    scz_i = []
    scz_jun = []
    # 水垂直右侧壁温
    scy_id = []
    scy_i = []
    scy_jun = []
    # 水垂直前侧壁温
    scq_id = []
    scq_i = []
    scq_jun = []
    # 水垂直后侧壁温
    sch_id = []
    sch_i = []
    sch_jun = []

    # 水螺旋左侧壁温
    slz_id = []
    slz_i = []
    slz_jun = []
    # 水螺旋右侧壁温
    sly_id = []
    sly_i = []
    sly_jun = []
    # 水螺旋前侧壁温
    slq_id = []
    slq_i = []
    slq_jun = []
    # 水螺旋后侧壁温
    slh_id = []
    slh_i = []
    slh_jun = []

    # 遍历水冷壁温数据
    # for i in range(len(column_headers1)):
    #     if '左侧墙垂直' in column_headers1[i]:
    #         scz_id.append(column_headers1[i])             # 添加所有左侧墙垂直对应的ID
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)  # 计算每一列均值
    #         scz_jun.append(datajun)                       # 存储所有均值
    #     elif '右侧墙垂直' in column_headers1[i]:
    #         scy_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         scy_jun.append(datajun)
    #     elif '前墙垂直' in column_headers1[i]:
    #         scq_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         scq_jun.append(datajun)
    #     elif '后墙垂直' in column_headers1[i]:
    #         sch_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         sch_jun.append(datajun)
    #
    #     elif '左侧墙螺旋' in column_headers1[i]:
    #         slz_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         slz_jun.append(datajun)
    #     elif '右侧墙螺旋' in column_headers1[i]:
    #         sly_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         sly_jun.append(datajun)
    #     elif '前墙螺旋' in column_headers1[i]:
    #         slq_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         slq_jun.append(datajun)
    #     elif '后墙螺旋' in column_headers1[i]:
    #         slh_id.append(column_headers1[i])
    #
    #         datajun = np.round(np.mean(dataa1[:, i]), 2)
    #         slh_jun.append(datajun)

    # 定义水冷壁温列表 存储二维数据
    cedscz = []
    cedscy = []
    cedscq = []
    cedsch = []
    cedslz = []
    cedsly = []
    cedslq = []
    cedslh = []

    # 给水冷壁温数据 添加二维数据
    # for i in range(len(scz_id)):
    #     a = i * 4 + 1
    #     # j = i + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'左侧墙垂直管'+'#'+guan,scz_jun[i]]  # a,a分别表示横纵坐标
    #     cedscz.extend([cedi])                            # extend表示合并列表
    #
    # for i in range(len(scy_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'右侧墙垂直管'+'#'+guan,scy_jun[i]]
    #     cedscy.extend([cedi])
    #
    # for i in range(len(scq_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'前墙垂直管'+'#'+guan,scq_jun[i]]
    #     cedscq.extend([cedi])
    #
    # for i in range(len(sch_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'后墙墙垂直管'+'#'+guan,sch_jun[i]]
    #     cedsch.extend([cedi])
    #
    # # 定义二维数据 (水冷壁螺旋段)
    # for i in range(len(slz_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'左侧墙螺旋管'+'#'+guan,slz_jun[i]]
    #     cedslz.extend([cedi])
    #
    # for i in range(len(sly_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'右侧墙螺旋管'+'#'+guan,sly_jun[i]]
    #     cedsly.extend([cedi])
    #
    # for i in range(len(slq_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'前墙螺旋管'+'#'+guan,slq_jun[i]]
    #     cedslq.extend([cedi])
    #
    # for i in range(len(slh_id)):
    #     a = i * 4 + 1
    #     guan = str(a) + "管"
    #     cedi = [a,a,'后墙螺旋管'+'#'+guan,slh_jun[i]]
    #     cedslh.extend([cedi])
    # print("水冷壁 数据:")
    # print(scz_id)
    # print(scz_jun)
    # print(cedscz)
    # print("know over")

    # choose_id='低温过热器上部屏出口段后墙横向'
    gao=[]
    gao2=[]
    di1=[]
    dii=[]
    di2=[]
    dii2=[]
    qian=[]
    hou=[]
    houu=[]
    zuo=[]
    you=[]
    for i in range(len(column_headers)):
        if '高温过热器' in column_headers[i]:
            gao.append(column_headers[i])
            shuzi=np.round(np.mean(dataa[:,i]),2)
            gao2.append(shuzi)
        elif '低温过热器上部屏出口段后墙' in column_headers[i]:
            di1.append(column_headers[i])
            shuzi=np.round(np.mean(dataa[:,i]),2)
            dii.append(shuzi)
        elif '低温过热器下部屏出口段后墙' in column_headers[i]:
            di2.append(column_headers[i])
            shuzi=np.round(np.mean(dataa[:,i]),2)
            dii2.append(shuzi)
        # elif '前墙' in column_headers[i]:
        #     qian.append(column_headers[i])
        elif '后墙隔墙' in column_headers[i]:
            hou.append(column_headers[i])
            shuzi=np.round(np.mean(dataa[:,i]),2)
            houu.append(shuzi)
        # elif '左墙' in column_headers[i]:
        #     zuo.append(column_headers[i])
        # elif '右墙' in column_headers[i]:
        #     you.append(column_headers[i])
        elif '左侧墙垂直' in column_headers[i]:
            scz_id.append(column_headers[i])             # 添加所有左侧墙垂直对应的ID
            datajun = np.round(np.mean(dataa[:, i]), 2)  # 计算每一列均值
            scz_jun.append(datajun)                       # 存储所有均值
        elif '右侧墙垂直' in column_headers[i]:
            scy_id.append(column_headers[i])

            datajun = np.round(np.mean(dataa[:, i]), 2)
            scy_jun.append(datajun)
        elif '前墙垂直' in column_headers[i]:
            scq_id.append(column_headers[i])

            datajun = np.round(np.mean(dataa[:, i]), 2)
            scq_jun.append(datajun)
        elif '后墙垂直' in column_headers[i]:
            sch_id.append(column_headers[i])

            datajun = np.round(np.mean(dataa[:, i]), 2)
            sch_jun.append(datajun)

        elif '左侧墙螺旋' in column_headers[i]:
            slz_id.append(column_headers[i])

            datajun = np.round(np.mean(dataa[:, i]), 2)
            slz_jun.append(datajun)
        elif '右侧墙螺旋' in column_headers[i]:
            sly_id.append(column_headers[i])

            datajun = np.round(np.mean(dataa[:, i]), 2)
            sly_jun.append(datajun)
        elif '前墙螺旋' in column_headers[i]:
            slq_id.append(column_headers[i])

            datajun = np.round(np.mean(dataa[:, i]), 2)
            slq_jun.append(datajun)
        elif '后墙螺旋' in column_headers[i]:
            slh_id.append(column_headers[i])
            datajun = np.round(np.mean(dataa[:, i]), 2)
            slh_jun.append(datajun)

    for i in range(len(scz_id)):
        a = i * 4 + 1
        # j = i + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),scz_jun[i],'左侧墙垂直管'+'#'+guan]  # a,a分别表示横纵坐标
        cedscz.extend([cedi])                            # extend表示合并列表

    for i in range(len(scy_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),scy_jun[i],'右侧墙垂直管'+'#'+guan]
        cedscy.extend([cedi])

    for i in range(len(scq_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),scq_jun[i],'前墙垂直管'+'#'+guan]
        cedscq.extend([cedi])

    for i in range(len(sch_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),sch_jun[i],'后墙墙垂直管'+'#'+guan]
        cedsch.extend([cedi])

    # 定义二维数据 (水冷壁螺旋段)
    for i in range(len(slz_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),slz_jun[i],'左侧墙螺旋管'+'#'+guan]
        cedslz.extend([cedi])

    for i in range(len(sly_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),sly_jun[i],'右侧墙螺旋管'+'#'+guan]
        cedsly.extend([cedi])

    for i in range(len(slq_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),slq_jun[i],'前墙螺旋管'+'#'+guan]
        cedslq.extend([cedi])

    for i in range(len(slh_id)):
        a = i * 4 + 1
        guan = str(a) + "管"
        cedi = [a,a,'#'+guan.replace('管','tube'),slh_jun[i],'后墙螺旋管'+'#'+guan]
        cedslh.extend([cedi])

    ced=[]
    ced2=[]
    ced3=[]
    ced4=[]
    for i in range(1,83):
        pai='第'+str(i)+'排'
        for j in range(len(gao)):
            if pai in gao[j]:
                for k in range(1,9):
                    guan=str(k)+'管'
                    if guan in gao[j]:
                        # print("s",gao[j])
                        # print(guan)
                        # print(pai+guan)
                        # cedi=[i,k,'高温过热器出口段横向'+pai+'#'+guan,gao2[j]]
                        cedi=[i,k,pai.replace('第','').replace('排','row')+'#'+guan.replace('管','tube'),gao2[j],'高温过热器出口段横向'+pai+'#'+guan]
                        ced.extend([cedi])

    for i in range(1,21):
        pai='第'+str(i)+'排'
        for j in range(len(di1)):
            if pai in di1[j]:
                for k in range(1,14):
                    guan='#'+str(k)+'管'
                    if guan in di1[j]:
                        # print(di1[j],i,j,k)
                        # print(pai+guan)
                        # cedi=[i,k,'低温过热器上部屏出口段后墙横向'+pai+guan,dii[j]]
                        cedi=[i,k,pai.replace('第','').replace('排','row')+'#'+guan.replace('管','tube'),dii[j],'低温过热器上部屏出口段后墙横向'+pai+guan]
                        ced2.extend([cedi])
    # print(np.array(ced2).shape)
    for i in range(1,21):
        pai='第'+str(i)+'排'
        for j in range(len(di2)):
            if pai in di2[j]:
                for k in range(13,14):
                    guan=str(k)+'管'
                    if guan in di2[j]:
                        # print(pai+guan)
                        cedi=[i,k,pai.replace('第','').replace('排','row')+'#'+guan.replace('管','tube'),dii2[j],'低温过热器下部屏出口段后墙横向'+pai+'#'+guan]
                        ced3.extend([cedi])

    for i in range(1,325):
        pai='#'+str(i)+'管'
        guan=str(i)+'排'
        for j in range(len(hou)):
            if pai in hou[j]:
                cedi=[i,5,'#'+guan.replace('管','tube'),houu[j],'后墙隔墙出口段#'+guan]
                ced4.extend([cedi])
    # c=[1,6,'none',houu[0]]
    # ced4.extend([c])
    #
    # c2=[1,14,'none',dii2[0]]
    # ced3.extend([c2])

    zhuz=[]
    zhuz2=[]
    zhuz3=[]
    zhuz4=[]
    for i in range(1,83):
        zhuzi=[i,1,9]
        zhuz.append(zhuzi)
    for i in range(1,21):
        zhuzi=[i,1,14]
        zhuz2.append(zhuzi)
    for i in range(1,21):
        zhuzi=[i,1,14]
        zhuz3.append(zhuzi)
    for i in range(1,324):
        zhuzi=[i,1,14]
        zhuz4.append(zhuzi)
    # print(zhuz4)
    # print(zhuz)
    # print(ced4)
    # print(ced)
    # print(max(gao2))
    # print(min(gao2))
    # print("choose_id",choose_id)
    if choose_id=='高温过热器出口段横向':
        ma=max(gao2)
        mi=min(gao2)
        data1=ced
        data2=zhuz
    elif choose_id=='低温过热器上部屏出口段后墙横向':
        ma=max(dii)
        mi=min(dii)
        data1=ced2
        data2=zhuz2
    elif choose_id=='低温过热器下部屏出口段后墙横向':
        ma=max(dii2)
        mi=min(dii2)
        data1=ced3
        data2=zhuz3
    elif choose_id=='后墙隔墙出口段':
        ma=max(houu)
        mi=min(houu)
        data1=ced4
        data2=zhuz4
    elif choose_id == '水冷壁左侧墙垂直':
        ma = max(scz_jun)
        mi = min(scz_jun)
        data1 = cedscz
    elif choose_id == '水冷壁右侧墙垂直':
        ma = max(scy_jun)
        mi = min(scy_jun)
        data1 = cedscy
    elif choose_id == '水冷壁前墙垂直':
        ma = max(scq_jun)
        mi = min(scq_jun)
        data1 = cedscq
    elif choose_id == '水冷壁后墙垂直':
        ma = max(sch_jun)
        mi = min(sch_jun)
        data1 = cedsch
    elif choose_id == '水冷壁左侧墙螺旋':
        ma = max(slz_jun)
        mi = min(slz_jun)
        data1 = cedslz
    elif choose_id == '水冷壁右侧墙螺旋':
        ma = max(sly_jun)
        mi = min(sly_jun)
        data1 = cedsly
    elif choose_id == '水冷壁前墙螺旋':
        ma = max(slq_jun)
        mi = min(slq_jun)
        data1 = cedslq
    elif choose_id == '水冷壁后墙螺旋':
        ma = max(slh_jun)
        mi = min(slh_jun)
        data1 = cedslh
    # print(data1)
    return data1,ma,mi


# 刷新页面 判断是否存在保存好的txt 直接可视化展示
@app.route('/shuaxin',methods=['GET','POST'])
def shuaxin():
    cluster_min = int(request.form.get('cluster_min'))
    cluster_max = int(request.form.get('cluster_max'))
    wenjian = request.form.get('wenjian')
    position = request.form.get('position')
    # print("wenjian", wenjian)

    # 打开所存储的文件
    path_out = "static/data/工况/"+wenjian+'/'+position+'('+str(cluster_min) +'~'+str(cluster_max)+')'+'.txt'
    print("path_out",path_out)

    try:
        b = open(path_out, "r", encoding='UTF-8')
        out = b.read()
        out = json.loads(out)
        print("文件存在")
        return jsonify({
            "all_score":out['all_score'],
            "cluster_name": out['cluster_name'],
            "erced":out['erced'],
            "max":out['max'],
            "min":out['min']
        })
    except IOError:
        return jsonify({
            "max":1
        })

@app.route('/Cluster_Compare',methods=['GET','POST'])
def Cluster_Compare():
    cluster_min=int(request.form.get('cluster_min'))
    cluster_max=int(request.form.get('cluster_max'))
    wenjian=request.form.get('wenjian')
    position=request.form.get('position')
    print("wenjian",wenjian)


    # 打开所存储的文件
    path_out = "static/data/工况/"+wenjian+'/'+position+'('+str(cluster_min) +'~'+str(cluster_max)+')'+'.txt'
    print("path_out",path_out)
    try:
        b = open(path_out, "r", encoding='UTF-8')
        out = b.read()
        out = json.loads(out)
        print("文件存在")
        return jsonify({
            "all_score":out['all_score'],
            "cluster_name": out['cluster_name'],
            "erced":out['erced'],
            "max":out['max'],
            "min":out['min']
        })
    except IOError:
        print("文件不存在")
        # 在截取ID前 先调用cedianwz函数 进行二维壁温数据可视化展示
        erced, ma, mi = cedianwz(position, wenjian)
        # print("erced",erced)

        # 截取字符串
        if "水冷壁" in position:
            position = position[3:]

        df = pd.read_csv('static/data/工况/' + wenjian + '/bw.csv')
        time = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        column_headers = df.columns.values
        df2 = pd.read_csv('static/data/juli3.csv')
        data2 = df2.iloc[:, :].values
        for i in range(data.shape[0]):
            for j in range(1, data.shape[1]):
                if math.isnan(data[i, j]):
                    data[i, j] = data[i - 1, j]

        index = []
        for i in range(len(column_headers)):
            if position in column_headers[i]:
                index.append(i - 1)
        data = data[:, index]
        data2 = data2[:, index]
        data2 = data2[index, :]

        # 返回计算好的所有聚类结果的轮廓系数
        score, cluster_name = cluster(data.T, data2, cluster_max, cluster_min)

        all_score = []
        for i in range(score.shape[0]):
            k = 2
            ce = []
            for j in range(score.shape[1]):
                ce.append([k, score[i, j]])
                k = k + 1
            all_score.extend([ce])

        # 将文件保存到txt文件中
        par = {
            "all_score": all_score,
            "cluster_name": cluster_name,
            "erced": erced,
            "max": ma,
            "min": mi,
        }
        # json.dumps将python数据结构转换为json格式
        c_list = json.dumps(par)
        a = open(r"static/data/工况/" + wenjian + '/' + position + '(' + str(cluster_min) + '~' + str(
            cluster_max) + ')' + '.txt', "w", encoding='UTF-8')
        a.write(c_list)
        a.close()

        return jsonify({
            "all_score":all_score,
            "cluster_name": cluster_name,
            "erced":erced,
            "max":ma,
            "min":mi
        })

import matplotlib.pyplot as plt

@app.route('/Clustering',methods=['GET','POST'])
def Clustering():
    cluster_name=request.form.get('cluster_name')
    cluster_num=int(request.form.get('cluster_num'))
    wenjian=request.form.get('wenjian')
    position=request.form.get('position')
    is_Fusion=int(request.form.get('is_Fusion'))
    print(cluster_name)
    print(cluster_num)
    print(wenjian)
    print(position)
    print(is_Fusion)

    position_new = []
    # 截取字符串ID
    if "水冷壁" in position:
        position_new = position[3:]
    else:
        position_new = position
    # print("position_new",position_new)

    df = pd.read_csv('static/data/工况/'+wenjian+'/bw.csv')
    time=df.iloc[:, 0].values
    data = df.iloc[:, 1:].values
    column_headers = df.columns.values
    column_headers=column_headers[1:]
    # print(column_headers)
    df2 = pd.read_csv('static/data/juli3.csv')
    data2=df2.iloc[:, :].values
    # print(data2.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if math.isnan(data[i,j]):
                data[i,j]=data[i-1,j]
    # print(data)
    # print(time)
    # print(data.shape)
    index=[]
    # cedian_name=[]
    for i in range(len(column_headers)):
        if position_new  in column_headers[i]:
            index.append(i)
            # cedian_name.append(column_headers[i])
    column_headers=column_headers[index]
    data=data[:,index]
    data2=data2[:,index]
    data2=data2[index,:]
    labels=cluster2(data.T,data2,cluster_name,cluster_num)
    # print("data2",data2.shape)
    print(labels)
    tsne=manifold.TSNE()
    a=tsne.fit_transform(data.T)  #进行数据降维,降成两维
    print("a.shape",a.shape)
    index=np.argwhere(labels==0).reshape(-1,)
    print(index)
    d=a[index,:]
    for i in range(d.shape[0]):
        plt.plot(d[i,0],d[i,1],'r.')

    index=np.argwhere(labels==1).reshape(-1,)
    d=a[index,:]
    for i in range(d.shape[0]):
        plt.plot(d[i,0],d[i,1],'b.')
    # plt.show()

    erced,ma,mi=cedianwz(position,wenjian)
    # print(labels)
    if is_Fusion==1:
        # zanshi=np.argwhere(labels==0).reshape(-1,)
        # ronghe=rongh_fusion(data,column_headers[zanshi],position)
        # ronghe=ronghe.reshape((-1,1))
        jilu=[]
        # jilu.append(position)
        ronghe=time.copy().reshape((-1,1))
        name=['time']
        for i in range(0,cluster_num):
            indexxx=np.argwhere(labels==i).reshape(-1,)
            jilu.append(len(indexxx))
            if len(indexxx)==1:
                ronghe1=data[:,indexxx]
            else:
                ronghe1=rongh_fusion(data[:,indexxx],column_headers[indexxx],position)
            ronghe1=ronghe1.reshape((-1,1))
            # print(i,ronghe.shape,ronghe1.shape)
            ronghe=np.hstack((ronghe,ronghe1))
            name.append('融合'+str(i+1))
        # print("ronghee",ronghe,ronghe.shape)
        ronghe=pd.DataFrame(ronghe,columns=name)
        if position=='高温过热器出口段横向':
            baocun_name='高温过热器'
        elif position=='低温过热器上部屏出口段后墙横向':
            baocun_name='低温过热器上部屏'
        elif position=='低温过热器下部屏出口段后墙横向':
            baocun_name='低温过热器下部屏'
        elif position=='后墙隔墙出口段':
            baocun_name='后墙隔墙'
        elif position=='水冷壁左侧墙垂直':
            baocun_name='水冷左垂'
        elif position == '水冷壁右侧墙垂直':
            baocun_name = '水冷右垂'
        elif position == '水冷壁前墙垂直':
            baocun_name = '水冷前垂'
        elif position == '水冷壁后墙垂直':
            baocun_name = '水冷后垂'
        elif position=='水冷壁左侧墙螺旋':
            baocun_name='水冷左螺'
        elif position == '水冷壁右侧墙螺旋':
            baocun_name = '水冷右螺'
        elif position == '水冷壁前墙螺旋':
            baocun_name = '水冷前螺'
        elif position == '水冷壁后墙螺旋':
            baocun_name = '水冷后螺'

        if cluster_name=='AgglomerativeCluster':
            baocun_cluster_name='AC'
        elif cluster_name=='AgglomerativeCluster    QA  _DTW':
            baocun_cluster_name='AC_DTW'
        elif cluster_name=='SpectralCluster':
            baocun_cluster_name='SC'
        elif cluster_name=='SpectralCluster_DTW':
            baocun_cluster_name='SC_DTW'
        else:
            baocun_cluster_name=cluster_name
        mingzi=baocun_name+'_'+baocun_cluster_name+'_'+str(cluster_num)
        jilu.insert(0,mingzi)
        ronghe.to_csv('static/data/工况/'+wenjian+'/'+baocun_name+'_'+baocun_cluster_name+'_'+str(cluster_num)+'.csv',index=None)
    else:
        jilu=0
    # print(labels.shape)
    labels=labels.tolist()
    for i in range(len(erced)):
        # print(i,erced[i])
        erced[i].append(labels[i])
        # print(erced[i])
    # if is_Fusion==1:
    #     ronghe=rongh_fusion(data,column,name)
    if is_Fusion==1:
        par={
            "cluster_name": cluster_name,
            "erced":erced,
            "max":ma,
            "min":mi,
            "cluster_num":cluster_num
        }
        c_list = json.dumps(par)
        a = open(r"static/data/工况/"+wenjian+'/'+baocun_name+'_'+baocun_cluster_name+'_'+str(cluster_num)+'.txt', "w",encoding='UTF-8')
        a.write(c_list)
        a.close()

    return jsonify({
        "cluster_name": cluster_name,
        "erced":erced,
        "max":ma,
        "min":mi,
        "jilu":jilu
    })

@app.route('/history_model',methods=['GET','POST'])
def history_model():
    # path = pathbase + "/static/ronghe_data"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    wenjian=request.form.get('wenjian')
    path = pathbase + "/static/data/工况/"+wenjian+'/'
    # print(path)
    listfile = os.listdir(path)  # 遍历一个文件夹下所有的文件
    files1 = []
    files2=[]
    print("listfile", listfile)
    for x in listfile:
        x = x.split('.')
        if x[1]!='txt' and x[0]!='bw' and x[0]!='qita':
            files1.append(x[0])
            name=x[0]
            name=name.replace('高温过热器','HTS')
            name=name.replace('低温过热器上部屏','LTS(Upper)')
            name=name.replace('低温过热器下部屏','LTS(Lower)')
            name=name.replace('后墙隔墙','Rear wall')
            name=name.replace('水冷前垂','VWW(Front)')
            name=name.replace('水冷后垂','VWW(Rear)')
            name=name.replace('水冷左垂','VWW(Left)')
            name=name.replace('水冷右垂','VWW(Right)')
            name=name.replace('水冷前螺','SWW(Front)')
            name=name.replace('水冷后螺','SWW(Rear)')
            name=name.replace('水冷左螺','SWW(Left)')
            name=name.replace('水冷右螺','SWW(Right)')
            files2.append(name)
    # print(files1)
    args = {'file':files1,'file2':files2}
    return jsonify(args)

@app.route('/dataset')
def dataset():
    # path = pathbase + "/static/ronghe_data"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    path = pathbase + "/static/data/工况/"
    # print(path)
    listfile = os.listdir(path)  # 遍历一个文件夹下所有的文件
    # files1 = []
    # os.path.getmtime() 函数是获取文件最后修改时间
    # os.path.getctime() 函数是获取文件最后创建时间
    # listfile = sorted(listfile,key=lambda x: os.path.getctime(os.path.join(path, x)))
    print("listfile", listfile)
    # for x in listfile:
    #     x = x.split('.')
    #     files1.append(x[0])
    # print(files1)
    args = {'file':listfile}
    return jsonify(args)

@app.route('/del_history_model',methods=['GET','POST'])
def del_history_model():
    wenjian=request.form.get('wenjian')
    file_name = request.form.get('file_name')
    path = pathbase + "/static/data/工况/"+wenjian+'/'
    path2=pathbase + "/static/data/工况/"+wenjian+'/'
    # print(path)
    # print(file_name)
    os.remove(path+file_name+'.csv')
    os.remove(path2+file_name+'.txt')
    # shutil.rmtree(path+file_name+'.csv')
    return 'success'

@app.route('/open_history_model')
def open_history_model():
    file_name = request.args.get('file_name')
    wenjian=request.args.get('wenjian')
    path_out = pathbase + "/static/data/工况/"+wenjian+'/'+file_name+'.txt'
    b = open(path_out, "r",encoding='UTF-8')
    out = b.read()
    out = json.loads(out)
    # print("----erced ----")
    # print(out['erced'])
    return jsonify({
        "cluster_name": out['cluster_name'],
        "erced":out['erced'],
        "max":out['max'],
        "min":out['min'],
        "cluster_num":out['cluster_num']
    })

@app.route('/CCA2',methods=['GET','POST'])
def CCA2():
    zuheX = request.form.get("zuheX")
    zuheX = json.loads(zuheX)
    zuheY = request.form.get("zuheY")
    zuheY = json.loads(zuheY)
    # print("zuhe",zuheX,zuheY)

    value=['高温过热器_AC_2','后墙隔墙_AC_2','低温过热器上部屏_SC_2','低温过热器下部屏_Birch_2']
    all_data=pd.read_csv('static/ronghe_data/'+value[0]+'.csv').iloc[:, 1:].values
    all_column=list(pd.read_csv('static/ronghe_data/'+value[0]+'.csv').columns.values[1:])
    for i in range(len(all_column)):
        all_column[i]=value[0]+'_'+all_column[i]
    # print(all_column)
    for i in range(1,len(value)):
        df = pd.read_csv('static/ronghe_data/'+value[i]+'.csv')
        column= list(df.columns.values[1:])
        for j in range(len(column)):
            column[j]=value[i]+'_'+column[j]
        data=df.iloc[:, 1:].values
        all_column=all_column+column
        # print(all_data.shape)
        # print(data.shape)
        all_data=np.hstack((all_data,data))
    qita_data=pd.read_csv("static/data/gongl.csv").iloc[:, 180:-2].values
    column2=pd.read_csv("static/data/gongl.csv").columns.values[ 180:-2]
    column1=np.array(all_column)
    # print("all_column",all_column)
    hebing_data=np.hstack((all_data,qita_data))
    for i in range(hebing_data.shape[0]):
        for j in range(hebing_data.shape[1]):
            if math.isnan(hebing_data[i,j]):
                hebing_data[i,j]=hebing_data[i-1,j]
    X_data=all_data[:,zuheX]
    Y_data=qita_data[:,zuheY]
    X=pd.DataFrame(X_data,columns=column1[zuheX])
    Y=pd.DataFrame(Y_data,columns=column2[zuheY])
    a,b,dcoef1,jieshi,jianyan,corr_xv,corr_yu,corr_xu,corr_yv=cca_as(X, Y)
    # print(dcoef1,zuheX,zuheY,column1[zuheX],column2[zuheY])
    jieshi=np.round(jieshi,3)
    # print(jianyan)
    # print(dcoef1)
    dcoef1=dcoef1.reshape((-1,1))
    jy=np.hstack((dcoef1,jianyan))
    # print(jy)
    # print(xishu)
    return jsonify({
        "jieshi": jieshi.tolist(),
        "jy":jy.tolist(),
        "a":a.T.tolist(),
        "b":b.T.tolist(),
        "n1":column1[zuheX].tolist(),
        "n2":column2[zuheY].tolist()
    })

@app.route('/CCA3',methods=['GET','POST'])
def CCA3():
    isIn = int(request.form.get('isIn'))
    zuheX = request.form.get("zuheX")
    zuheX = json.loads(zuheX)
    zuheY = request.form.get("zuheY")
    zuheY = json.loads(zuheY)
    # print("zuhe",zuheX,zuheY)

    value=['高温过热器_AC_2','后墙隔墙_AC_2','低温过热器上部屏_SC_2','低温过热器下部屏_Birch_2']
    all_data=pd.read_csv('static/ronghe_data/'+value[0]+'.csv').iloc[:, 1:].values
    all_column=list(pd.read_csv('static/ronghe_data/'+value[0]+'.csv').columns.values[1:])
    for i in range(len(all_column)):
        all_column[i]=value[0]+'_'+all_column[i]
    # print(all_column)
    for i in range(1,len(value)):
        df = pd.read_csv('static/ronghe_data/'+value[i]+'.csv')
        column= list(df.columns.values[1:])
        for j in range(len(column)):
            column[j]=value[i]+'_'+column[j]
        data=df.iloc[:, 1:].values
        all_column=all_column+column
        # print(all_data.shape)
        # print(data.shape)
        all_data=np.hstack((all_data,data))
    qita_data=pd.read_csv("static/data/gongl.csv").iloc[:, 180:-2].values
    for i in range(qita_data.shape[0]):
        for j in range(qita_data.shape[1]):
            if math.isnan(qita_data[i,j]):
                qita_data[i,j]=qita_data[i-1,j]
    column2=pd.read_csv("static/data/gongl.csv").columns.values[ 180:-2]
    column1=np.array(all_column)
    # print("all_column",all_column)
    hebing_data=np.hstack((all_data,qita_data))
    for i in range(hebing_data.shape[0]):
        for j in range(hebing_data.shape[1]):
            if math.isnan(hebing_data[i,j]):
                hebing_data[i,j]=hebing_data[i-1,j]
    rongyu=[]
    zuida=[]
    if isIn==1:
        X_data=all_data[:,zuheX]
        X=pd.DataFrame(X_data,columns=column1[zuheX])
        for i in range(len(zuheY)):
            Y_data=qita_data[:,zuheY[i]]
            Y=pd.DataFrame(Y_data,columns=column2[zuheY[i]])
            a,b,dcoef1,jieshi,jianyan,corr_xv,corr_yu,corr_xu,corr_yv=cca_as(X, Y)
            # print(dcoef1,round(2,dcoef1[0]))
            result1=0
            result2=0
            for j in range(len(dcoef1)):
                result1=result1+jieshi[0,j]*dcoef1[j]*dcoef1[j]
                result2=result2+jieshi[3,j]*dcoef1[j]*dcoef1[j]
            zuida.append(round(result1,2))
            rongyu.append(round(result2,2))
            # zuida.append(round(dcoef1[0],2))
            # rongyu.append(round(0.5*(jieshi[0,0]+jieshi[3,0])*dcoef1[0]*dcoef1[0],2))
    else:
        Y_data=qita_data[:,zuheY]
        Y=pd.DataFrame(Y_data,columns=column2[zuheY])
        for i in range(len(zuheX)):
            X_data=all_data[:,zuheX[i]]
            X=pd.DataFrame(X_data,columns=column1[zuheX[i]])
            a,b,dcoef1,jieshi,jianyan,corr_xv,corr_yu,corr_xu,corr_yv=cca_as(X, Y)
            result1=0
            result2=0
            for j in range(len(dcoef1)):
                result1=result1+jieshi[0,j]*dcoef1[j]*dcoef1[j]
                result2=result2+jieshi[3,j]*dcoef1[j]*dcoef1[j]
            zuida.append(round(result1,2))
            rongyu.append(round(result2,2))
            # zuida.append(round(jieshi[0,0]*dcoef1[0]*dcoef1[0],2))
            # rongyu.append(round(jieshi[3,0]*dcoef1[0]*dcoef1[0],2))
            # zuida.append(round(dcoef1[0],2))
            # rongyu.append(round(0.5*(jieshi[0,0]+jieshi[3,0])*dcoef1[0]*dcoef1[0],2))
    var1=np.round(np.var(all_data, axis = 0),2)
    var2=np.round(np.var(qita_data, axis = 0),2)
    # print("var1",var1)
    # print(var2)
    return jsonify({
        "zuida": zuida,
        "rongyu":rongyu,
        "max1":max(zuida),
        "max2":max(rongyu),
        "var1":var1.tolist(),
        "var2":var2.tolist()
    })

@app.route('/CoCluster3',methods=['GET','POST'])
def CoCluster3():
    Row_Cluster_num = int(request.form.get('Row_Cluster_num'))
    Col_Cluster_num = int(request.form.get('Col_Cluster_num'))
    value = request.form.get("value")
    value = json.loads(value)
    listt = request.form.get("listt")
    listt = json.loads(listt)
    listt_1 = request.form.get("listt_1")
    listt_1 = json.loads(listt_1)
    listt2 = request.form.get("listt2")
    listt2 = json.loads(listt2)
    listt2_1 = request.form.get("listt2_1")
    listt2_1 = json.loads(listt2_1)
    # print("listt2_1",listt2_1,listt_1)
    # print("value",listt,listt2,value,Row_Cluster_num,Col_Cluster_num)
    value=['高温过热器_AC_2','后墙隔墙_AC_2','低温过热器上部屏_SC_2','低温过热器下部屏_Birch_2']
    all_data=pd.read_csv('static/ronghe_data/'+value[0]+'.csv').iloc[:, 1:].values
    all_column=list(pd.read_csv('static/ronghe_data/'+value[0]+'.csv').columns.values[1:])
    for i in range(len(all_column)):
        all_column[i]=value[0]+'_'+all_column[i]
    # print(all_column)
    for i in range(1,len(value)):
        df = pd.read_csv('static/ronghe_data/'+value[i]+'.csv')
        column= list(df.columns.values[1:])
        for j in range(len(column)):
            column[j]=value[i]+'_'+column[j]
        data=df.iloc[:, 1:].values
        all_column=all_column+column
        # print(all_data.shape)
        # print(data.shape)
        all_data=np.hstack((all_data,data))
    qita_data=pd.read_csv("static/data/gongl.csv").iloc[:, 180:-2].values
    column2=pd.read_csv("static/data/gongl.csv").columns.values[ 180:-2]
    column1=np.array(all_column)
    # print("all_column",all_column)
    hebing_data=np.hstack((all_data,qita_data))
    for i in range(hebing_data.shape[0]):
        for j in range(hebing_data.shape[1]):
            if math.isnan(hebing_data[i,j]):
                hebing_data[i,j]=hebing_data[i-1,j]
    corr_ma=np.corrcoef(hebing_data.T)
    Similarity_ma=corr_ma[0:all_data.shape[1],all_data.shape[1]:]
    du1 = open('static/result/c结果.txt', "r",encoding='UTF-8')
    out1 = du1.read()
    out1 = json.loads(out1)
    du2 = open('static/result/c组合.txt', "r",encoding='UTF-8')
    out2 = du2.read()
    out2 = json.loads(out2)
    jieguo=np.array(out1['jieguo'])
    X=out2['X']
    Y=out2['Y']
    # print(X)
    # print(Y)
    index1=[]
    for i in range(len(X)):
        # d = [False for c in listt if c not in X[i]]
        # print("X",listt,X[i])
        same_values = set(listt_1) & set(X[i])
        if set(listt) <= set(X[i]) and len(X[i])<=Row_Cluster_num and len(same_values)==0:
            index1.append(i)
    index2=[]
    for i in range(len(Y)):
        # print("Y",listt2,Y[i])
        # d = [False for c in listt if c not in Y[i]]
        same_values = set(listt2_1) & set(Y[i])
        # print("same_values",same_values,listt2_1,Y[i])
        if set(listt2) <= set(Y[i]) and len(Y[i])<=Col_Cluster_num and len(same_values)==0:
            print("Y2",listt2,Y[i])
            index2.append(i)
    # print(index1,index2)
    zuheX=np.array(X)[index1].tolist()
    zuheY=np.array(Y)[index2].tolist()
    # print("jieguo",jieguo.shape,index1,index2)
    jieguo=jieguo[index1,:]
    jieguo=jieguo[:,index2]
    corr_pj=[]
    cluster_xgx=[]
    name1=[]
    name2=[]
    for i in range(len(index1)):
        cluster_pj1=[]
        cluster_xgx1=[]
        name1.append(column1[zuheX[i]].tolist())
        X_data=all_data[:,zuheX[i]]
        for j in range(len(index2)):
            if i==0:
                name2.append(column2[zuheY[j]].tolist())
            # print(Similarity_ma.shape)
            # Y_data=qita_data[:,zuheY[j]]
            # X=pd.DataFrame(X_data,columns=column1[zuheX[i]])
            # Y=pd.DataFrame(Y_data,columns=column2[zuheY[j]])
            # # print(X)
            # # print(Y)
            # a,b,dcoef1,jieshi,jianyan=cca_as(X, Y)
            # print(jieguo[i,j],dcoef1,zuheX[i],zuheY[j],column1[zuheX[i]],column2[zuheY[j]])
            # cluster_xgx1.append(float(round(dcoef1[0],3)))
            xgx1=abs(Similarity_ma[:,zuheY[j]])
            xgx2=abs(xgx1[zuheX[i],:])
            # print(zuheX[i],zuheX[i],xgx2.shape)
            he=np.sum(xgx2)
            he=round(he/(len(zuheX[i])*len(zuheY[j])),3)
            # print("平均相关性: ",he)
            cluster_pj1.append(he)
        corr_pj.append(cluster_pj1)
        # cluster_xgx.append(cluster_xgx1)
    data_len=[]
    data_len.append(all_data.shape[1])
    data_len.append(qita_data.shape[1])
    data_len=np.array(data_len)
    return jsonify({
        "cluster1": zuheX,
        "cluster2": zuheY,
        "cluster_xgx":jieguo.tolist(),
        "cluster_pj":corr_pj,
        "data_len":data_len.tolist(),
        "biwen_column":column1.tolist(),
        "qita_column":column2.tolist(),
        "name1":name1,
        "name2":name2
    })

@app.route('/Switch',methods=['GET','POST'])
def Switch():
    duibi1=request.form.get('duibi1')
    duibi1 = np.array(json.loads(duibi1))
    duibi2=request.form.get('duibi2')
    duibi2 = np.array(json.loads(duibi2))
    duibi3=request.form.get('duibi3')
    duibi3 = np.array(json.loads(duibi3))
    mingzi=request.form.get('mingzi')
    mingzi = json.loads(mingzi)
    print("duibi1",duibi1)
    print("duibi2",duibi2)
    print("duibi3",duibi3)
    d=np.vstack((duibi1,duibi2))
    d=np.vstack((d,duibi3))
    tsne = manifold.TSNE(n_components=2,init='pca',random_state=501,perplexity=3)
    # tsne = manifold.TSNE(n_components=2,init='pca',random_state=501)
    # tsne =manifold.LocallyLinearEmbedding(2)
    dd = tsne.fit_transform(d)
    x_min, x_max = dd.min(0), dd.max(0)
    X_norm = (dd - x_min) / (x_max - x_min)
    d1=X_norm[0:len(duibi1),:]
    d2=X_norm[len(duibi1):len(duibi1)*2,:]
    d3=X_norm[len(duibi1)*2:,:]
    # d1=dd[0:len(duibi1),:]
    # d2=dd[len(duibi1):len(duibi1)*2,:]
    # d3=dd[len(duibi1)*2:,:]
    # d1 = tsne.fit_transform(duibi1)
    # d2 = tsne.fit_transform(duibi2)
    # d3 = tsne.fit_transform(duibi3)
    print(d1,d1.shape)
    print(d2,d2.shape)
    print(d3,d3.shape)
    d1=d1.tolist()
    d2=d2.tolist()
    d3=d3.tolist()
    print("d1 list",d1)
    print("d2 list",d2)
    print("d3 list",d3)

    allmwlist = []
    for i in range(len(d1)):
        mwlist = []

        mwlist.append(d1[i])
        mwlist.append(d2[i])
        mwlist.append(d3[i])

        allmwlist.append(mwlist)



    print("allmwlist",allmwlist)
    print("lenall",len(allmwlist))
    # for i in range(len(mwlist)):
    #     if(i % 3 == 0):


    max1 = []
    max2 = []
    max3 = []
    for i in range(len(d1)):
        max1.append(max(d1[i]))
        max2.append(max(d2[i]))
        max3.append(max(d3[i]))
    print("max1",max1)
    print("max2",max2)
    print("max3",max3)

    allmax = []
    allmin = []

    for i in range(len(max1)):
        allmax.append(max1[i])
        allmax.append(max2[i])
        allmax.append(max3[i])

    mindata = min(allmax)
    maxdata = max(allmax) + 300
    print("mindata",mindata)
    print("maxdata",maxdata)

    for i in range(len(d1)):
        d1[i].append(mingzi[i])
        d2[i].append(mingzi[i])
        d3[i].append(mingzi[i])


    return jsonify({
        "d1": d1,
        "d2": d2,
        "d3": d3,
        # 'mindata':-1,
        # 'maxdata':2,
        'mindata':mindata,
        'maxdata':1,
        'allmwlist':allmwlist
    })
# np.set_printoptions(threshold=None)
@app.route('/CoCluster4',methods=['GET','POST'])
def CoCluster4():
    wenjian = request.form.get('wenjian')
    Relevance_sel = int(request.form.get('Relevance_sel'))
    Row_Cluster_num = int(request.form.get('Row_Cluster_num'))
    Col_Cluster_num = int(request.form.get('Col_Cluster_num'))
    sample = int(request.form.get('sample'))
    value = request.form.get("value")
    value = json.loads(value)
    listt = request.form.get("listt")
    listt = json.loads(listt)
    listt2 = request.form.get("listt2")
    listt2 = json.loads(listt2)
    print("---wenjian",wenjian)
    print("---Relevance_sel",Relevance_sel)
    print("---Row_Cluster_num",Row_Cluster_num)
    print("---Col_Cluster_num",Col_Cluster_num)
    print("---value",value)
    print("---listt",listt)
    print("---listt2",listt2)
    # print("value",listt,listt2,value,Row_Cluster_num,Col_Cluster_num)
    path = pathbase + "/static/data/工况/"+wenjian+'/'
    print("---path",path)
    listfile = os.listdir(path)
    print("---listfile",listfile)
    value3 = []
    for x in listfile:
        x = x.split('.')
        if x[1]!='txt' and x[0]!='bw' and x[0]!='qita':
            value3.append(x[0])
    # value2=['高温过热器_AC_2','后墙隔墙_AC_2','低温过热器上部屏_SC_2','低温过热器下部屏_Birch_2', '水冷前垂_AC_2', '水冷后垂_AC_3', '水冷左垂_AC_2', '水冷右垂_SC_2', '水冷前螺_AC_2', '水冷后螺_SC_2', '水冷右螺_AC_2', '水冷左螺_AC_2']
    value2=['高温过热器','后墙隔墙','低温过热器上部屏','低温过热器下部屏', '水冷前垂', '水冷后垂', '水冷左垂', '水冷右垂', '水冷前螺', '水冷后螺', '水冷右螺', '水冷左螺']
    value = []
    while value2:
        nn = value2.pop(0)
        for i in range(len(value3)):
            if nn in value3[i]:
                value.append(value3[i])
                break
    print("value",value)

    # 先读取一个文件壁温数据
    all_data = pd.read_csv(path + value[0]+'.csv').iloc[:, 1:].values
    all_column = list(pd.read_csv(path+value[0]+'.csv').columns.values[1:])
    for i in range(len(all_column)):
        all_column[i]=value[0]+'_'+all_column[i]
    print(all_column)

    # 再依次读取其他文件的壁温数据
    for i in range(1,len(value)):
        df = pd.read_csv(path+value[i]+'.csv')
        column= list(df.columns.values[1:])
        for j in range(len(column)):
            column[j]=value[i]+'_'+column[j]
        data = df.iloc[:, 1:].values
        all_column = all_column+column

        # all_data 存储的是所有的壁温数据
        all_data = np.hstack((all_data,data))

    print("all_data.shape",all_data.shape)

    # 读取影响参数数据
    if wenjian=='MW(595~604)_(1 15_43_45~1 16_52_20)_824':
        qita_data = pd.read_csv(path+"qita.csv",encoding='gb18030').iloc[:, 177:-2].values
        column2 = pd.read_csv(path+"qita.csv",encoding='gb18030').columns.values[177:-2]
        MW = pd.read_csv(path+"qita.csv",encoding='gb18030').iloc[:, -2].values
        for i in range(MW.shape[0]):
            if math.isnan(MW[i]):
                MW[i]=MW[i-1]
        times = pd.read_csv(path+"qita.csv",encoding='gb18030').iloc[:, 0].values
        time = times.tolist()
    else:
        qita_data = pd.read_csv(path+"qita.csv").iloc[:, 177:-2].values
        column2 = pd.read_csv(path+"qita.csv").columns.values[177:-2]
        MW = pd.read_csv(path+"qita.csv").iloc[:, -2].values
        for i in range(MW.shape[0]):
            if math.isnan(MW[i]):
                MW[i]=MW[i-1]
        times = pd.read_csv(path+"qita.csv").iloc[:, 0].values
        time = times.tolist()

    timenew = []

    for i in range(len(time)):
        a = time[i]
        # print("---",a)
        timenew.append(a[10:])
    MW2=[]
    # for i in range(MW.shape[0]):
    #     MW2.append([times[i],MW[i]])
    # ti=pd.read_csv(path+"qita.csv").iloc[:, -2].values
    print("column2",column2,column2.shape)
    column1=np.array(all_column)
    print("all_column",all_column)
    hebing_colum= all_column + column2.tolist()
    print("---alldata.shape[1]",all_data.shape[1])
    print(hebing_colum[all_data.shape[1]:])

    # 蒋壁温数据和参数数据合并为总体的数据
    hebing_data = np.hstack((all_data,qita_data))

    # 进行缺失值处理
    for i in range(hebing_data.shape[0]):
        for j in range(hebing_data.shape[1]):
            if math.isnan(hebing_data[i,j]):
                hebing_data[i,j]=hebing_data[i-1,j]
                # print("hebing",hebing_data[i,j],i,j)
    corr_ma = np.corrcoef(hebing_data.T)

    nnn = ['省煤器进口给水温度', '过热度', '启动分离器壁温高值', '过热器二级减温器出口温度A1', '过热器二级减温器出口温度B1', '一次再热再热器事故喷水减温器进口温度A', '一次再热再热器事故喷水减温器出口温度A1', '一次再热高温再热器出口蒸汽温度A1', '一次再热再热器事故喷水减温器进口温度B', '一次再热再热器事故喷水减温器出口温度B1', '一次再热高温再热器出口蒸汽温度B1', '二次再热再热器事故喷水减温器进口温度A', '二次再热再热器事故喷水减温器出口温度A1', '二次再热再热器微量喷水减温器进口温度A1', '二次再热再热器微量喷水减温器出口温度A1', '二次再热高温再热器出口蒸汽温度A2', '二次再热再热器事故喷水减温器进口温度B', '二次再热再热器事故喷水减温器出口温度B1', '二次再热再热器微量喷水减温器进口温度B1', '二次再热再热器微量喷水减温器出口温度B1', '二次再热高温再热器出口蒸汽温度B2', '高温过热器出口主蒸汽压力']
    jilu = []
    print("---hebing_colum",hebing_colum)
    for j in range(all_data.shape[1],corr_ma.shape[0]):
            if hebing_colum[j] not in nnn:
                jilu.append(j)

    # 删除不需要的列和数据
    jilu = sorted(jilu,reverse=True)
    print("---jilu",jilu)
    for i in range(len(jilu)):
        hebing_colum.pop(jilu[i])
    print(len(hebing_colum))
    print(hebing_data.shape)

    # 删除hebing_data中不需要的数据
    hebing_data = np.delete(hebing_data, jilu, axis=1)


    # 计算壁温和参数之间的皮尔逊相关性系数
    corr_ma2 = np.corrcoef(hebing_data.T)

    print("hebing_data",hebing_data)
    print("hebing_data.shape",hebing_data.shape)
    # pd.DataFrame(hebing_data).to_csv('hebing_data.csv')
    # pd.DataFrame(corr_ma2).to_csv('corr_ma2.csv')
    for i in range(corr_ma2.shape[0]):
        for j in range(corr_ma2.shape[1]):
            if math.isnan(corr_ma2[i,j]):
                print("hebing2",corr_ma2[i,j])
    Similarity_ma = corr_ma2[0:all_data.shape[1],all_data.shape[1]:]
    print("corr_ma2",corr_ma2)
    print("corr_ma2.shape",corr_ma2.shape)


    X=pd.DataFrame(hebing_data[:,:all_data.shape[1]],columns=hebing_colum[:all_data.shape[1]])
    Y=pd.DataFrame(hebing_data[:,all_data.shape[1]:],columns=hebing_colum[all_data.shape[1]:])


    a,b,dcoef1,jieshi,jianyan,corr_xv,corr_yu,corr_xu,corr_yv=cca_as(X, Y)
    biwen=X.values
    biwen -= np.mean(biwen,axis=0)
    biwen /= np.std(biwen,axis=0)
    biwen=biwen.dot(a)
    qita=Y.values
    qita -= np.mean(qita,axis=0)
    qita /= np.std(qita,axis=0)
    qita=qita.dot(b)
    he=np.hstack((biwen,qita))
    # he1 = pd.DataFrame(he)
    # temmpp = he1.corr()
    # print("temmpp",temmpp)
    qita2=[]
    biwen2=[]
    for i in range(MW.shape[0]):
        MW2.append([timenew[i],MW[i]])
        # biwen2_1=[]
        # qita2_1=[]
        # for j in range(8):
        #     biwen2_1.append([times[i],biwen[i,j]])
        #     qita2_1.append([times[i],qita[i,j]])
        # qita2.append(qita2_1)
        # biwen2.append(biwen2_1)
    for j in range(8):
        biwen2_1=[]
        qita2_1=[]
        for i in range(MW.shape[0]):
            biwen2_1.append([timenew[i],biwen[i,j]])
            qita2_1.append([timenew[i],qita[i,j]])
        qita2.append(qita2_1)
        biwen2.append(biwen2_1)
        # print("biwen2",biwen2)
    # print("biwen",biwen)
    # print("qita",qita)
    # print("ccc",np.corrcoef(he.T).shape)
    # print("ccc",np.corrcoef(he.T)[22:,:22])
    tu=[]
    for i in range(biwen.shape[1]):
        tux=[]
        for j in range(0,biwen.shape[0],5):
            tux.append([biwen[j][i],qita[j][i]])
        tu.append(tux)

    aa = biwen.shape[0]
    print("数据集长度:",aa)

    tua = len(tu[0])-1
    print("tua",tua)

    for i in range(0,8):
        dataulist = []
        datavlist = []
        for j in range(0,tua):
            datau = tu[i][j][0]
            dataulist.append(datau)
            datav = tu[i][j][1]
            datavlist.append(datav)

        # print("dataulist",dataulist)
        # print("datavlist",datavlist)

        # 找到散点图中 u 最大的值的下标
        datamaxu = max(dataulist)
        dataminu = min(dataulist)
        print("----")
        print(datamaxu)
        print(dataminu)
        indexmax = dataulist.index(datamaxu)
        indexmin = dataulist.index(dataminu)

        print("indexmax",indexmax)
        print("indexmin",indexmin)

        # U下标所对应的V
        maxv = datavlist[indexmax]
        minv = datavlist[indexmin]

        print("maxv",maxv)
        print("minv",minv)

        if(maxv - minv < 0):
            dcoef1[i] = -dcoef1[i]

    print("new dcoef1",dcoef1)

    jianyan_corr=[]
    for i in range(8):
        k=0
        j=1
        jianyan_corr2=[]
        while k+sample<qita.shape[0]:
            # print("jianyan_corr",j,biwen[k:k+sample,0],qita[k:k+sample,0])
            jianyan_corr2.append([j,np.corrcoef(biwen[k:k+sample,i],qita[k:k+sample,i])[0,1],(np.var(biwen[k:k+sample,i])+np.var(qita[k:k+sample,i]))/2,np.var(biwen[k:k+sample,i]),np.var(qita[k:k+sample,i]),np.mean(biwen[k:k+sample,i]),np.mean(qita[k:k+sample,i])])
            k+=sample
            j+=1
        jianyan_corr.append(jianyan_corr2)
    # print("jianyan_corr",jianyan_corr)
    a=a[:,0:8]
    b=b[:,0:8]
    corr_xv=corr_xv[:,0:8]
    corr_yu=corr_yu[:,0:8]
    corr_xu=corr_xu[:,0:8]
    corr_yv=corr_yv[:,0:8]
    jieshi=jieshi[:,0:8]
    # print(a)
    # print(b)
    # print("dcoef1",dcoef1)
    # print(corr_xv,corr_yu)
    # print(jieshi)
    # print(jianyan)
    if Relevance_sel==1:
        d1=corr_xv
        d2=corr_yu
    elif Relevance_sel==2:
        d1=corr_xu
        d2=corr_yv
    else:
        d1=a
        d2=b
    bi_index=np.argsort(-abs(d1),axis=0)[:4,:3].T
    qita_index=np.argsort(-abs(d2),axis=0)[:4,:3].T
    biwen_colunm2=hebing_colum[:all_data.shape[1]]
    # qita_column2=hebing_colum[all_data.shape[1]:]
    for i in range(len(biwen_colunm2)):
        biwen_colunm2[i]=biwen_colunm2[i].replace('高温过热器','HTS')
        biwen_colunm2[i]=biwen_colunm2[i].replace('融合','FT')
        biwen_colunm2[i]=biwen_colunm2[i].replace('后墙隔墙','Rear Wall')
        biwen_colunm2[i]=biwen_colunm2[i].replace('低温过热器下部屏','LTS(Upper)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('低温过热器上部屏','LTS(Lower)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷前垂','VMM(Front)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷后垂','VMM(Rear)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷左垂','VMM(Left)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷右垂','VMM(Right)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷前螺','VMM(Front)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷后螺','VMM(Rear)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷左螺','VMM(Left)')
        biwen_colunm2[i]=biwen_colunm2[i].replace('水冷右螺','VMM(Right)')
    qita_column2=pd.read_csv('static/data/被影响参数.csv',encoding='gb18030').iloc[:, 1].values
    print(qita_column2)
    print("hebing-data",hebing_data)
    return jsonify({
        "biwen":biwen2,
        "qita":qita2,
        "biwen2":biwen[::5,:8].tolist(),
        "qita2":qita[::5,:8].tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
        "jieshi": np.round(jieshi,3).tolist(),
        "biwen_column":hebing_colum[:all_data.shape[1]],
        "qita_column":hebing_colum[all_data.shape[1]:],
        "biwen_column2":biwen_colunm2,
        "qita_column2":qita_column2.tolist(),
        "dcoef1":np.round(dcoef1,3).tolist(),
        "corr_xv":d1.tolist(),
        "corr_yu":d2.tolist(),
        "corr_xv2":d1.T.tolist(),
        "corr_yu2":d2.T.tolist(),
        "Similarity_ma":np.round(Similarity_ma,2).tolist(),
        "corr":corr_ma2.tolist(),
        "nn":corr_xv.shape[1],
        "tu":tu,
        "jianyan_corr":jianyan_corr,
        "bi_index":bi_index.tolist(),
        "qita_index":qita_index.tolist(),
        "MW":MW.tolist(),
        # "times":times.tolist(),
        "time":timenew,
        "times":timenew,
        "MW2":MW2,
        "hebing_data":hebing_data.tolist()
    })


@app.route('/Cocluster4_2',methods=['GET','POST'])
def Cocluster4_2():
    tongji = request.form.get("tongji")
    tongji = json.loads(tongji)
    print("tongji",tongji)

    save_data = request.form.get("save_data")
    save_data = json.loads(save_data)

    all_data = np.array(save_data[0])

    # 遍历保存的矩阵数据 并转化为numpy格式 并纵向合并数据
    for i in range(1,len(save_data)):
        savedata = np.array(save_data[i])
        all_data = np.vstack((all_data,savedata))

    print("new all_data",all_data)
    print("new all_data.shape",all_data.shape)

    xgx = np.corrcoef(all_data.T)
    print("--- xgx ---")
    print(xgx)
    print("xgx.shape",xgx.shape)


    lista = [6,6,6]
    return jsonify({
        "lista":lista,
        "xgx":xgx.tolist()
    })

@app.route('/CoCluster2',methods=['GET','POST'])
def CoCluster2():
    Row_Cluster_num = int(request.form.get('Row_Cluster_num'))
    Col_Cluster_num = int(request.form.get('Col_Cluster_num'))
    value = request.form.get("value")
    value = json.loads(value)
    listt = request.form.get("listt")
    listt = json.loads(listt)
    listt2 = request.form.get("listt2")
    listt2 = json.loads(listt2)
    print("value",listt,listt2,value,Row_Cluster_num,Col_Cluster_num)
    value=['高温过热器_AC_2','后墙隔墙_AC_2','低温过热器上部屏_SC_2','低温过热器下部屏_Birch_2']
    all_data=pd.read_csv('static/ronghe_data/'+value[0]+'.csv').iloc[:, 1:].values
    all_column=list(pd.read_csv('static/ronghe_data/'+value[0]+'.csv').columns.values[1:])
    for i in range(len(all_column)):
        all_column[i]=value[0]+'_'+all_column[i]
    print(all_column)
    for i in range(1,len(value)):
        df = pd.read_csv('static/ronghe_data/'+value[i]+'.csv')
        column= list(df.columns.values[1:])
        for j in range(len(column)):
            column[j]=value[i]+'_'+column[j]
        data=df.iloc[:, 1:].values
        all_column=all_column+column
        print(all_data.shape)
        print(data.shape)
        all_data=np.hstack((all_data,data))
    qita_data=pd.read_csv("static/data/gongl.csv").iloc[:, 180:-2].values
    column2=pd.read_csv("static/data/gongl.csv").columns.values[ 180:-2]
    column1=np.array(all_column)
    print("all_column",all_column)
    hebing_data=np.hstack((all_data,qita_data))
    for i in range(hebing_data.shape[0]):
        for j in range(hebing_data.shape[1]):
            if math.isnan(hebing_data[i,j]):
                hebing_data[i,j]=hebing_data[i-1,j]
    corr_ma=np.corrcoef(hebing_data.T)
    Similarity_ma=corr_ma[0:all_data.shape[1],all_data.shape[1]:]
    du1 = open('static/result/c结果.txt', "r",encoding='UTF-8')
    out1 = du1.read()
    out1 = json.loads(out1)
    du2 = open('static/result/c组合.txt', "r",encoding='UTF-8')
    out2 = du2.read()
    out2 = json.loads(out2)
    jieguo=np.array(out1['jieguo'])
    X=out2['X']
    Y=out2['Y']
    print(X)
    print(Y)
    index1=[]
    for i in range(len(X)):
        if len(X[i])==Row_Cluster_num:
            index1.append(i)
    index2=[]
    for i in range(len(Y)):
        if len(Y[i])==Col_Cluster_num:
            index2.append(i)
    zuheX=np.array(X)[index1].tolist()
    zuheY=np.array(Y)[index2].tolist()
    print("jieguo",jieguo.shape,index1,index2)
    jieguo=jieguo[index1,:]
    jieguo=jieguo[:,index2]
    corr_pj=[]
    cluster_xgx=[]
    name1=[]
    name2=[]
    for i in range(len(index1)):
        cluster_pj1=[]
        cluster_xgx1=[]
        name1.append(column1[zuheX[i]].tolist())
        X_data=all_data[:,zuheX[i]]
        for j in range(len(index2)):
            if i==0:
                name2.append(column2[zuheY[j]].tolist())
            print(Similarity_ma.shape)
            # Y_data=qita_data[:,zuheY[j]]
            # X=pd.DataFrame(X_data,columns=column1[zuheX[i]])
            # Y=pd.DataFrame(Y_data,columns=column2[zuheY[j]])
            # # print(X)
            # # print(Y)
            # a,b,dcoef1,jieshi,jianyan=cca_as(X, Y)
            # print(jieguo[i,j],dcoef1,zuheX[i],zuheY[j],column1[zuheX[i]],column2[zuheY[j]])
            # cluster_xgx1.append(float(round(dcoef1[0],3)))
            xgx1=abs(Similarity_ma[:,zuheY[j]])
            xgx2=abs(xgx1[zuheX[i],:])
            print(zuheX[i],zuheX[i],xgx2.shape)
            he=np.sum(xgx2)
            he=round(he/(len(zuheX[i])*len(zuheY[j])),3)
            print("平均相关性: ",he)
            cluster_pj1.append(he)
        corr_pj.append(cluster_pj1)
        # cluster_xgx.append(cluster_xgx1)
    data_len=[]
    data_len.append(all_data.shape[1])
    data_len.append(qita_data.shape[1])
    data_len=np.array(data_len)
    return jsonify({
        "cluster1": zuheX,
        "cluster2": zuheY,
        "cluster_xgx":jieguo.tolist(),
        "cluster_pj":corr_pj,
        "data_len":data_len.tolist(),
        "biwen_column":column1.tolist(),
        "qita_column":column2.tolist(),
        "name1":name1,
        "name2":name2,
        "corr":corr_ma.tolist()
    })

@app.route('/CoCluster',methods=['GET','POST'])
def CoCluster():
    Row_Cluster_num = int(request.form.get('Row_Cluster_num'))
    Col_Cluster_num = int(request.form.get('Col_Cluster_num'))
    value = request.form.get("value")
    value = json.loads(value)
    print("value",value,Row_Cluster_num,Col_Cluster_num)
    all_data=pd.read_csv('static/ronghe_data/'+value[0]+'.csv').iloc[:, 1:].values
    all_column=list(pd.read_csv('static/ronghe_data/'+value[0]+'.csv').columns.values[1:])
    for i in range(len(all_column)):
        all_column[i]=value[0]+'_'+all_column[i]
    print(all_column)
    for i in range(1,len(value)):
        df = pd.read_csv('static/ronghe_data/'+value[i]+'.csv')
        column= list(df.columns.values[1:])
        for j in range(len(column)):
            column[j]=value[i]+'_'+column[j]
        data=df.iloc[:, 1:].values
        all_column=all_column+column
        print(all_data.shape)
        print(data.shape)
        all_data=np.hstack((all_data,data))
    qita_data=pd.read_csv("static/data/gongl.csv").iloc[:, 180:-2].values
    column2=pd.read_csv("static/data/gongl.csv").columns.values[ 180:-2]
    column1=np.array(all_column)
    print("all_column",all_column)
    hebing_data=np.hstack((all_data,qita_data))
    for i in range(hebing_data.shape[0]):
        for j in range(hebing_data.shape[1]):
            if math.isnan(hebing_data[i,j]):
                hebing_data[i,j]=hebing_data[i-1,j]
    corr_ma=np.corrcoef(hebing_data.T)
    Similarity_ma=corr_ma[0:all_data.shape[1],all_data.shape[1]:]
    model = SpectralBiclustering(n_clusters=[Row_Cluster_num,Col_Cluster_num],random_state=0)
    model.fit(Similarity_ma)
    cluster_xgx=[]
    cluster_pj=[]
    cluster1=[]
    cluster2=[]
    data_len=[]
    data_len.append(all_data.shape[1])
    data_len.append(qita_data.shape[1])
    data_len=np.array(data_len)
    name1=[]
    name2=[]
    for i in range(0,Row_Cluster_num):
        index1=np.argwhere(model.row_labels_==i).reshape(-1,)
        X_data=all_data[:,index1]
        cluster1.append(index1.tolist())
        cluster_xgx1=[]
        cluster_pj1=[]
        name1.append(column1[index1].tolist())
        for j in range(0,Col_Cluster_num):
            print(i,j)
            index2=np.argwhere(model.column_labels_==j).reshape(-1,)
            if i==0:
                cluster2.append(index2.tolist())
                name2.append(column2[index2].tolist())
            # biwen_data=biwen_data[:,index1]
            # print(index2)
            Y_data=qita_data[:,index2]
            X=pd.DataFrame(X_data,columns=column1[index1])
            Y=pd.DataFrame(Y_data,columns=column2[index2])
            # print(X)
            # print(Y)
            a,b,dcoef1,jieshi,jianyan,corr_xv,corr_yu,corr_xu,corr_yv=cca_as(X, Y)
            print(dcoef1,column1[index1],column2[index2])
            cluster_xgx1.append(float(round(dcoef1[0],3)))
            xgx1=abs(Similarity_ma[:,index2])
            xgx2=abs(xgx1[index1,:])
            print(xgx2.shape)
            he=np.sum(xgx2)
            he=round(he/(index1.shape[0]*index2.shape[0]),3)
            print("平均相关性: ",he)
            cluster_pj1.append(he)
        cluster_pj.append(cluster_pj1)
        cluster_xgx.append(cluster_xgx1)
    print(cluster_xgx)
    print(cluster1)
    print(cluster2)
    return jsonify({
        "cluster1": cluster1,
        "cluster2": cluster2,
        "cluster_xgx":cluster_xgx,
        "cluster_pj":cluster_pj,
        "data_len":data_len.tolist(),
        "biwen_column":column1.tolist(),
        "qita_column":column2.tolist(),
        "name1":name1,
        "name2":name2
    })


def cluster(data,data2,cluster_max,cluster_min):
    Similarity_ma=np.corrcoef(data)
    print("Similarity_ma",Similarity_ma)
    model_SpectralCoclustering = []
    SCoC_silhouette_score = []

    model_KMeans = []
    KMeans_silhouette_score = []

    model_AgglomerativeClustering = []
    AC_silhouette_score = []

    model_AgglomerativeClustering_DTW = []
    ACDTW_silhouette_score = []

    model_SpectralClustering = []
    SC_silhouette_score = []

    model_Birch = []
    Birch_silhouette_score = []

    model_KMedoids_DTW=[]
    KMDTW_silhouette_score = []

    model_KMedoids=[]
    KM_silhouette_score = []

    model_SpectralClustering_DTW = []
    SCDTW_silhouette_score = []

    print("data2",data2.shape)
    for i in range(cluster_min,cluster_max):
        print("i",i)
        num_clusters = i #聚类数
        # model1 = SpectralCoclustering(n_clusters=num_clusters, random_state=0)
        # model1.fit(Similarity_ma)
        # model_SpectralCoclustering.append(model1)
        # values1 = silhouette_score(Similarity_ma,model1.row_labels_)
        # SCoC_silhouette_score.append(values1)
        model1 = AgglomerativeClustering(n_clusters=num_clusters,linkage='average',affinity='precomputed')
        model1.fit(data2)
        model_AgglomerativeClustering_DTW.append(model1)
        values1 = silhouette_score(data2,model1.labels_)
        ACDTW_silhouette_score.append(values1)
        print(0)

        model2 = KMeans(n_clusters=num_clusters, random_state=0)
        model2.fit(data)
        model_KMeans.append(model2)
        values2 = silhouette_score(data,model2.labels_)
        KMeans_silhouette_score.append(values2)
        # print(model2.labels_.shape)
        print(1)

        model3 = AgglomerativeClustering(n_clusters=num_clusters)
        model3.fit(Similarity_ma)
        model_AgglomerativeClustering.append(model3)
        values3 = silhouette_score(Similarity_ma,model3.labels_)
        AC_silhouette_score.append(values3)
        print(2)

        model4 = SpectralClustering(n_clusters=num_clusters)
        model4.fit(Similarity_ma)
        model_SpectralClustering.append(model4)
        values4 = silhouette_score(Similarity_ma,model4.labels_)
        SC_silhouette_score.append(values4)
        print(model4.labels_.shape)
        print(3)

        model5 = Birch(n_clusters=num_clusters)
        model5.fit(data)
        model_Birch.append(model5)
        values5 = silhouette_score(data,model5.labels_)
        Birch_silhouette_score.append(values5)
        print(4)

        model6 = KMedoids(n_clusters=num_clusters,random_state=0,metric="precomputed")
        y_pred =model6.fit(data2)
        model_KMedoids_DTW.append(model6)
        print(5.1)
        values6 = silhouette_score(data2,model6.labels_)
        KMDTW_silhouette_score.append(values6)
        print(5)

        model7 = KMedoids(n_clusters=num_clusters,random_state=0,metric="euclidean")
        y_pred2 =model7.fit(data)
        model_KMedoids.append(model7)
        values7 = silhouette_score(data,model7.labels_)
        KM_silhouette_score.append(values7)
        print(6)

        # model8 = SpectralClustering(n_clusters=num_clusters,affinity='precomputed')
        # model8.fit(data2)
        # model_SpectralClustering_DTW.append(model8)
        # values8 = silhouette_score(data2,model8.labels_)
        # SCDTW_silhouette_score.append(values8)
        # print(7)

    # SCoC_silhouette_score = np.reshape(SCoC_silhouette_score,(-1,1))
    KM_silhouette_score = np.reshape(KM_silhouette_score,(-1,1))
    AC_silhouette_score = np.reshape(AC_silhouette_score,(-1,1))
    SC_silhouette_score = np.reshape(SC_silhouette_score,(-1,1))
    Birch_silhouette_score = np.reshape(Birch_silhouette_score,(-1,1))
    KMDTW_silhouette_score = np.reshape(KMDTW_silhouette_score,(-1,1))
    # SCDTW_silhouette_score = np.reshape(SCDTW_silhouette_score,(-1,1))
    ACDTW_silhouette_score = np.reshape(ACDTW_silhouette_score,(-1,1))
    KMeans_silhouette_score=np.reshape(KMeans_silhouette_score,(-1,1))
    all_silhouette_score=KM_silhouette_score
    all_silhouette_score=np.hstack((all_silhouette_score,KMDTW_silhouette_score))
    all_silhouette_score=np.hstack((all_silhouette_score,KMeans_silhouette_score))
    all_silhouette_score=np.hstack((all_silhouette_score,AC_silhouette_score))
    all_silhouette_score=np.hstack((all_silhouette_score,ACDTW_silhouette_score))
    all_silhouette_score=np.hstack((all_silhouette_score,SC_silhouette_score))
    # all_silhouette_score=np.hstack((all_silhouette_score,SCDTW_silhouette_score))
    all_silhouette_score=np.hstack((all_silhouette_score,Birch_silhouette_score))
    all_silhouette_score=np.round(all_silhouette_score,2)
    cluster_name=['KM','KM_DTW','KMeans','AC','AC_DTW','SC','Birch']
    return all_silhouette_score.T,cluster_name

def cluster2(data,data2,cluster_name,cluster_num):
    Similarity_ma=np.corrcoef(data)
    print(Similarity_ma.shape)
    if cluster_name=='KMedoids':
        model = KMedoids(n_clusters=cluster_num,random_state=0,metric="euclidean")
        model.fit(data)
    elif cluster_name=='KMedoids_DTW':
        model = KMedoids(n_clusters=cluster_num,random_state=0,metric="precomputed")
        model.fit_predict(data2)
    elif cluster_name=='KMeans':
        model = KMeans(n_clusters=cluster_num, random_state=0)
        model.fit(data)
    elif cluster_name=='AgglomerativeCluster':
        model = AgglomerativeClustering(n_clusters=cluster_num)
        model.fit(Similarity_ma)
    elif cluster_name=='AgglomerativeCluster_DTW':
        model = AgglomerativeClustering(n_clusters=cluster_num,linkage='average',affinity='precomputed')
        model.fit(data2)
    elif cluster_name=='SpectralCluster':
        model = SpectralClustering(n_clusters=cluster_num)
        model.fit(Similarity_ma)
    elif cluster_name=='Birch':
        model = Birch(n_clusters=cluster_num)
        model.fit(data)



    cluster_name=['KM','KM_DTW','KMeans','AC','AC_DTW','SC','Birch']
    return model.labels_

def batch(yizu,erzu):
    print(yizu)
    print(erzu)
    a = np.mean(yizu,axis=1)
    b = np.mean(erzu,axis=1)
    print("1",a,b)
    yizu=yizu.astype(float)
    erzu=erzu.astype(float)
    c = np.std(yizu, ddof=1,axis=1)
    print(c.shape)
    print(c)
    d = np.std(erzu, ddof=1,axis=1)
    print("d",d)
    if yizu.shape[1]==1:
        dd=np.hstack((yizu,erzu))
        c=np.std(dd, ddof=1,axis=1)
        d=np.std(dd, ddof=1,axis=1)
        # c=d.copy()
    elif erzu.shape[1]==1:
        dd=np.hstack((yizu,erzu))
        c=np.std(dd, ddof=0,axis=1)
        d=np.std(dd, ddof=0,axis=1)
    T=rh(a,b,c,d)
    T=T.reshape((1,-1))
    return  T
def rh(a,b,c,d):
    t=(((d**2)*a)+((c**2)*b))/(c**2+d**2)
    return t
def fenzu1(columns_,ji,ou):
    for i in range(len(columns_)):
        #提取字符串中的数字
        pl=re.findall(r"\d+",columns_[i])
        if int(pl[0])%2==0:
            ou.append(i)
        elif int(pl[0])%2!=0:
            ji.append(i)
    return ji,ou
def fenzu2(columns_,ji,ou):
    for i in range(len(columns_)):
        #提取字符串中的数字
        pl=re.findall(r"\d+",columns_[i])
        if int(pl[1])%2==0:
            ou.append(i)
        elif int(pl[1])%2!=0:
            ji.append(i)
    return ji,ou
def fenzu3(columns_,ji,ou):
    for i in range(len(columns_)):
        #提取字符串中的数字
        if int(i)%2==0:
            ou.append(i)
        elif int(i)%2!=0:
            ji.append(i)
    return ji,ou
def rongh_fusion(data,column,name):
    ji=[]
    ou=[]
    if '后墙' in name:
        ji,ou=fenzu1(column,ji,ou)
        print(column)
        print("sda",ji,ou)
        if len(ji)==0 or len(ou)==0:
            ji=[]
            ou=[]
            ji,ou=fenzu3(column,ji,ou)
        ronghe1=batch(data[:,ji],data[:,ou])
        # ji,ou=fenzu3(column,ji,ou)
        # print("2",ji,ou)
        # ronghe1=batch(data[:,ji],data[:,ou])
        # print("s",ronghe1)
        # ronghe=np.vstack((time,ronghe))
        # ronghe=pd.DataFrame(ronghe.T,columns=['时间','融合'])
    else:
        ji,ou=fenzu1(column,ji,ou)
        print("sda",ji,ou)
        if len(ji)==0 or len(ou)==0:
            ji=[]
            ou=[]
            ji,ou=fenzu3(column,ji,ou)
        ronghe1=batch(data[:,ji],data[:,ou])
        print(ronghe1)
        # ji,ou=fenzu2(column,ji,ou)
        # ronghe2=batch(data[:,ji],data[:,ou])
        # if len(ji)==0 or len(ou)==0:
        #     ji=[]
        #     ou=[]
        #     ji,ou=fenzu3(column,ji,ou)
        # print(ronghe2)
        # ronghe=np.vstack((time,ronghe1))
        # ronghe=np.vstack((ronghe,ronghe2))
        print(ronghe1.shape)
    return ronghe1
        # ronghe=pd.DataFrame(ronghe.T,columns=['时间','排融合','列融合'])
    # ronghe.to_csv('static/ronghe_data/'+name+'.csv',index=None)

def mse(y_true, y_pred):
    """
    均方误差  用 真实值-预测值 然后平方之后求和平均
    """

    n = len(y_true)
    mse = sum(np.square(x - y)for x,y in zip(y_true,y_pred)) / n
    mse = round(mse,2)


    return mse

def rmse(y_true, y_pred):
    """
    均方根误差
    """

    n = len(y_true)
    rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(y_true, y_pred)) / n)
    rmse = round(rmse,2)

    return rmse

def mae(y_true, y_pred):
    """
    MAE 平均绝对误差
    """

    n = len(y_true)
    sum = 0
    for i in range(n):
        sum = sum + abs(y_true[i] - y_pred[i])
    mae = sum / n
    mae = round(mae,2)

    return mae


def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape 平均绝对百分比误差
    """

    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    mape = round(mape,2)

    return mape

# def use_open_url(port):
#     # browser_path=r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe'
#     # #这里的‘firefox'只是一个浏览器的代号，可以命名为自己认识的名字，只要浏览器路径正确
#     # web.register('google chrome', web.Mozilla('mozilla'), web.BackgroundBrowser(browser_path))
#     # #web.get('firefox').open(url,new=1,autoraise=True)
#     # web.get('google chrome').open_new_tab(url)
#     # print('use_chrome_open_url open url ending ....')
#     sleep(10)
#     webbrowser.open('http://127.0.0.1:4060/')
    # web.open_new_tab("http://127.0.0.1:"+ str(port) + "/")

import webbrowser
import threading

if __name__ == '__main__':
    # webbrowser.open_new('http://127.0.0.1:4060/')
    # threading.Timer(1.50, lambda: webbrowser.open('http://127.0.0.1:4060/')).start()
    # use_open_url(4060)
    app.run(port=4060, debug=True, threaded=True)
    # webbrowser.open_new('http://127.0.0.1:4060/')
    # webbrowser.open_new('http://127.0.0.1:4060/')
    # web.open_new_tab("http://127.0.0.1:4060/")
    # webbrowser.open('http://127.0.0.1:4060/')