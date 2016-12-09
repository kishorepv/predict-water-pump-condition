# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:05:15 2016

@author: shrieko (Kishore P. V.)
"""

import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import collections
import scipy.stats
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.ensemble import ExtraTreesClassifier
import itertools
import os
from sklearn import tree
from io import StringIO
from IPython.display import Image
import pydotplus
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
import statistics



#plt.rcParams["figure.figsize"]=(14,5)
def preprocess(data, quant, cat):
    data["amount_tsh"]=data["amount_tsh"]
    data["gps_height"]=data["gps_height"].where(data["gps_height"]>0, other=0)
    data["longitude"]=data["longitude"]
    data["latitude"]=data["latitude"]
    data["basin"]=data["basin"].apply(to_lower)
    data["lga"]=data["lga"].apply(to_lower)
    data["scheme_management"]=data["scheme_management"].apply(to_lower)
    #data["scheme_management"]=data["scheme_management"].apply(recode_nan)
    lst=list()
    for ele in data["scheme_management"]:
        if type(ele)==type(float()) and np.isnan(ele):
            lst.append("unknown")
        else:
            lst.append(ele)
    data["scheme_management"]=lst
    data["extraction_type"]=data["extraction_type"].apply(to_lower)
    data["management"]=data["management"].apply(to_lower)
    data["management_group"]=data["management_group"].apply(to_lower)
    data["payment"]=data["payment"].apply(to_lower)
    data["water_quality"]=data["water_quality"].apply(to_lower)
    data["quantity"]=data["quantity"].apply(to_lower)
    data["source"]=data["source"].apply(to_lower)
    data["waterpoint_type"]=data["waterpoint_type"].apply(to_lower)
    for cat_var in cat:
        data[cat_var] = data[cat_var].astype("category")
    return data

def process2(data):
    recodings={"status_group":{"functional":2, "functional needs repair":1, "non functional":0}}
    recoding_features=["basin", "lga", "scheme_management", "extraction_type", "management"
    ,"management_group", "payment", "water_quality", "quantity", "source", "waterpoint_type"
    ]

    for feature in recoding_features:
        categories=sorted(data[feature].unique())
        mapping=collections.defaultdict(int)
        for index,categ in enumerate(categories):
            mapping[categ]=index+1
        recodings[feature]=mapping
        data[feature]= data[feature].map(mapping)


    data["installer"]=data["installer"].apply(recode_installer)
    data["installer"]=data["installer"].apply(recode_installer2)
    return data, recodings, recoding_features


def plot_confusion_matrix(cm, classes,cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def describe(var):
    d={"amount_tsh":"Amount of water available", "gps_height":"Altitude of the well", "population":"Population around the well", "latitude":"latitude", "longitude":"longitude",
       "construction_year":"Year of construction", "basin":"Geographic water basin",  "region_code":"Region code", "district_code":"District code",
    "lga":"lga",  "scheme_management":   "Scheme Management","extraction_type":"Water Extraction type", "management":"Management", "management_group":"Management Group",
    "payment":"Payment for pump water", "water_quality":"Water quality", "quantity":"Quantity of water available", "source":"Water source",
    "waterpoint_type":"Waterpoint type", "installer":"Pump Installer", "status_group":"Water pump status"
        }
    return  d[var].strip().upper()
    return ' '.join([word.upper() for word in var.split('_')])
    """
    cat_xlabels=["Year of construction", "Basin", "Region code", "District code", "Lga",
    "Scheme Management","Extraction type","Management","Management Group", "Payment for water","Water quality","Quantity", "Source","Waterpoint type", "Installer of water pump"]+[ "Status group"]
    """
    """
    quant=["amount_tsh", "gps_height","population"]+["longitude","latitude"]
    cat=["construction_year", "basin", "region_code", "district_code", "lga",
    "scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source",
                 "waterpoint_type", "installer"]
    """

def recode_installer(ele):
        if type(ele)==type(float()) and np.isnan(ele):
            return "unknown"
        elif "known" in ele:
            return "other"
        elif ele=="0":
            return "unknown"
        elif ele=="-":
            return "unknown"
        elif type(ele)==type(str()):
            return ele.lower()

def recode_installer2(ele):
    c=ele[0]
    if ele=="unknown":
        return 0
    elif ord('a')<=ord(c)<ord('z'):
        return ord(c)-ord('a')
    else:
        return 0

def to_lower(ele):
    if type(ele)==type(str()):
        return ele.lower()
    else:
        return ele
def recode_nan(ele):
    if type(ele)==type(float()) and np.isnan(ele):
        return "unknown"
    else:
        return ele



if __name__=="__main__":
    fname="train4910797b-ee55-40a7-8668-10efd5c1b960.csv"
    fname2="train_labels0bf8bc6e-30d0-4c50-956a-603fc693d966.csv"
    ftest1="test702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv"
    ans_file="SubmissionFormat.csv"
    ans_format=pd.read_csv(ans_file)
    dataset=pd.read_csv(fname)
    output=pd.read_csv(fname2)
    test_data=pd.read_csv(ftest1)
    dataset["status_group"]=output["status_group"].copy()
    quant=["amount_tsh", "gps_height","population"]+["longitude","latitude"]
    cat=["construction_year", "basin", "region_code", "district_code", "lga",
    "scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source",
                 "waterpoint_type", "installer"]
    out= ["status_group"]
    my_features=cat+quant+out
    data=dataset[my_features].copy()
    test_data=test_data[quant+cat+["id"]].copy()
    id1=test_data["id"].copy()
    test_data=preprocess(test_data, quant, cat)
    data=preprocess(data, quant, cat)
    data[out[0]] = data[out[0]].astype("category")

    print("\nTable 1: Quantitative variables summary\n")
    df_quant=pd.DataFrame()
    for feat in quant:
        des=data[feat].describe()
        df_quant=df_quant.append(des)
    del df_quant["count"]
    df_quant.index=[describe(feature) for feature in df_quant.index]
    tbl1=tabulate(df_quant, headers="keys", tablefmt="psql")
    print(tbl1)

    print("\nTable 2: Categorical variables summary\n")
    df_cat=pd.DataFrame()
    for feat in cat+out:
        des=data[feat].describe()
        df_cat=df_cat.append(des)
    df_cat.index=[describe(feature) for feature in df_cat.index]
    tb2=tabulate(df_cat, headers="keys", tablefmt="psql")
    print(tb2)

    print("\nVISUALIZATIONS:\n")
    print("Categorical variables:\n")
    cat_plot1=["construction_year", "basin", "region_code", "district_code",
    "scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source",
                 "waterpoint_type"]
    out_plot1= ["status_group"]
    for feature in cat_plot1+out_plot1:
        seaborn.countplot(x=feature, data=data)
        plt.xlabel(describe(feature))
        plt.title("Univariate plot for {}".format(describe(feature)))
        plt.xticks(rotation=45)
        plt.show()

    print("Quantitative variables:\n")
    quant_plot1=["amount_tsh", "gps_height","population"]+["longitude","latitude"]
    for feature in quant_plot1:
        seaborn.distplot(data[feature], kde=False)
        plt.xlabel(describe(feature))
        plt.title("Univariate plot for {}".format(describe(feature)))
        plt.xticks(rotation=45)
        plt.show()
    recode_output={"functional":2, "functional needs repair":1, "non functional":0}
    data[out[0]]= data[out[0]].map(recode_output)
    data[out[0]] = pd.to_numeric(data[out[0]], errors="coerce")

    print("\nBivariate Plots:\n")
    quant_plot2=["amount_tsh", "gps_height","population"]+["longitude","latitude"]
    cat_plot2=["construction_year", "basin", "region_code", "district_code", "lga",
    "scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source",
                 "waterpoint_type", "installer"]
    out_plot2= ["status_group"]

    for feature in quant_plot2:
        plot_data=data[[feature]+out].copy()
        out_to_binary={0:0, 1:1, 2:1}
        plot_data=plot_data[plot_data[feature]!=0].copy()
        plot_data[feature]=pd.qcut(plot_data[feature], 4, labels=["25th%tile","50%tile","75%tile","100%tile"])
        plot_data[out[0]]=plot_data[out[0]].map(out_to_binary)
        seaborn.factorplot(x=feature, y=out[0], data=plot_data, kind="bar", ci=None)
        #seaborn.regplot(x=feature, y=out[0], fit_reg=False, data=plot_data)
        plt.xticks(rotation=45)
        plt.xlabel(describe(feature))
        plt.ylabel(describe(out_plot2[0]))
        plt.title("{} vs {}".format(describe(out_plot2[0]), describe(feature)))
        plt.show()
    cat_plot21=[ "basin", "region_code", "district_code",
     "scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source",
                  "waterpoint_type"]
    cat_plot22=["construction_year"]
    for feature in cat_plot21+cat_plot22:
        plot_data=data[[feature]+out_plot2].copy()
        out_to_binary={0:0, 1:0, 2:1}
        if feature in cat_plot22:
            plot_data=plot_data[plot_data[feature]!=0].copy()
            plot_data[feature]=pd.qcut(plot_data[feature], 4, labels=["25th%tile","50%tile","75%tile","100%tile"])
        plot_data[out[0]]=plot_data[out[0]].map(out_to_binary)
        seaborn.factorplot(x=feature, y=out[0], data=plot_data, kind="bar", ci=None)
        plt.xticks(rotation=45)
        plt.xlabel(describe(feature))
        plt.ylabel(describe(out[0]))
        plt.title("{} vs {}".format(describe(out[0]), describe(feature)))
        plt.show()
    data, recodings, recoding_features= process2(data)
    test_data, recodingsx, recoding_featuresx= process2(test_data)
    catx=[ "basin", "region_code", "district_code", "lga"
    ,"scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source"
    ,"waterpoint_type"]
    data2=data.copy()

    print("\nTable 3: Chi Square Test\n")
    df_chisq=pd.DataFrame()
    row=dict()
    for feature in cat+quant:
        if feature in quant:
            data2=data2[data2[feature]!=0].copy()
            data2[feature]=pd.qcut(data2[feature], 4, labels=["25th%tile","50%tile","75%tile","100%tile"])
        data2[feature]=data2[feature].astype("category")
        out_to_binary={0:0, 1:0, 2:1}
        data2["output"]=data2[out[0]].map(out_to_binary)
        ct=pd.crosstab(data2["output"], data2[feature])
        cs=scipy.stats.chi2_contingency(ct)
        row["chi-square value"]=cs[0]
        row["p value"]=cs[1]
        df_chisq=df_chisq.append(pd.Series(row, name=describe(feature)))
    tbl3=tabulate(df_chisq, headers="keys", tablefmt="psql")
    print(tbl3)
    quant_tree=["amount_tsh", "gps_height","population"]+["longitude","latitude"]
    cat_tree=["construction_year", "basin", "region_code", "district_code"
    ,"scheme_management","extraction_type","management","management_group", "payment","water_quality","quantity", "source",
                 "waterpoint_type", "installer"]

    print("\nClassification Tree:\n")
    feature_names=cat_tree+quant_tree
    predictors=data[feature_names]
    test_predictors=test_data[feature_names]
    target = data.status_group
    pred_train,pred_test,tar_train,tar_test=train_test_split(predictors, target, test_size=.4)
    classifier=DecisionTreeClassifier()
    classifier=classifier.fit(pred_train,tar_train)
    predictions=classifier.predict(pred_test)
    cmatrix=sklearn.metrics.confusion_matrix(tar_test,predictions, labels=[0,1,2])
    cacc=sklearn.metrics.accuracy_score(tar_test, predictions)
    print("Accuracy of classification (by classification tree): ",np.round(cacc,3))
    cmatrix_norm=cmatrix.astype('float')/cmatrix.sum(axis=1)[:, np.newaxis]
    df_cmat=pd.DataFrame()
    indices=["non functional" ,"functional needs repair", "functional"]
    for index, row in zip(indices, cmatrix_norm):
        df_cmat[index]=np.round(row,2)
    df_cmat.index=indices

    print("\nTable 4: Normalized Confusion Matrix (Classification Tree)\n")
    tbl4=tabulate(df_cmat, headers="keys", tablefmt="psql")
    print(tbl4)

    """
    os.chdir("C:\TREES")
    #Displaying the decision tree
    out_str = StringIO()
    tree.export_graphviz(classifier, out_file=out_str)

    print("above")
    graph=pydotplus.graph_from_dot_data(out_str.getvalue())
    print("below")
    with open("img.png", 'wb') as f:
        f.write(graph.create_png())
    """

    print("\nRandom Forest:\n")
    prediction_sets=list()
    accuracy_sets=list()
    for i in range(1,28):
        classifier_rf=RandomForestClassifier(n_estimators=i)
        classifier_rf=classifier_rf.fit(pred_train,tar_train)
        predictions_rf=classifier_rf.predict(pred_test)
        ans_predictions_rf=classifier_rf.predict(test_predictors)
        prediction_sets.append(ans_predictions_rf)
        cacc_rf=sklearn.metrics.accuracy_score(tar_test, predictions_rf)
        accuracy_sets.append(cacc_rf)
    def ensemble_ans(topper_ans, not_ans, t):
        if topper_ans!=not_ans:
            return topper_ans
        else:
            for v,k in t[::-1]:
                if k!=not_ans:
                    return k
    ans=list()
    d=collections.defaultdict(int)
    mx_index=np.argmax(accuracy_sets)
    for index,tup in enumerate(zip(*prediction_sets)):
        for x in tup:
            d[x]+=1
        t=sorted([(b,a) for a,b in d.items()])
        not_ans=(t[0][1])
        topper_ans=prediction_sets[mx_index][index]
        ans.append(ensemble_ans(topper_ans, not_ans, t))
    print(accuracy_sets)

    maps={v:k for k,v in recodings["status_group"].items()}
    def remap(val):
        return maps[val]
    s=pd.Series(ans, name="status_group")
    s=s.apply(remap)
    print(len(s), len(s.dropna()))
    out_df=pd.DataFrame()
    out_df["id"]=test_data["id"]
    out_df["status_group"]=s
    out_df.to_csv("ans_ensemble_rf.csv", index=False)

    cmatrix2=sklearn.metrics.confusion_matrix(tar_test,predictions, labels=[0,1,2])
    cmatrix2_norm=cmatrix2.astype('float')/cmatrix2.sum(axis=1)[:, np.newaxis]
    df_cmat2=pd.DataFrame()
    indices=["non functional" ,"functional needs repair", "functional"]
    for index, row in zip(indices, cmatrix2_norm):
        df_cmat2[index]=np.round(row,2)
    df_cmat2.index=indices
    print("\n Table 5: Normalized Confusion Matrix (Random Forest)\n")
    tbl5=tabulate(df_cmat2, headers="keys", tablefmt="psql")
    print(tbl5)


    acc2=sklearn.metrics.accuracy_score(tar_test, predictions)
    print("Acc: ", acc2)
    model = ExtraTreesClassifier()
    model.fit(pred_train,tar_train)
    feature_importance=model.feature_importances_
    lst=list()
    for feature, imp in zip(feature_names, feature_importance):
        lst.append((imp, feature))
    lst=sorted(lst)[::-1]
    df_randforest=pd.DataFrame()
    for imp, feature in lst:
        d=dict()
        d["Relative Importance"]=np.round(imp, 2)
        df_randforest=df_randforest.append(pd.Series(d, name=describe(feature)))
    print("\nTable 6: Relative Importance of features\n")
    tbl7=tabulate(df_randforest, headers="keys", tablefmt="psql")
    print(tbl7)

    trees=range(25)
    accuracy=np.zeros(25)
    for idx in range(len(trees)):
        classifier=RandomForestClassifier(n_estimators=idx + 1)
        classifier=classifier.fit(pred_train,tar_train)
        predictions=classifier.predict(pred_test)
        accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
    print(accuracy[19:])
    plt.cla()
    plt.plot(trees, accuracy)
    plt.xlabel("Number of Trees")
    plt.ylabel("Prediction Accuracy")
    plt.title("Prediction Accuracy vs Number of Trees")
    plt.show()
    
