# 附加图4投稿修订资料

最终文件为Supplementary_Fig4.pptx，单页180 × 230 mm左右，原生可编辑矢量对象；英文Times New Roman，文字黑色。PDF由兼容渲染器导出，本环境缺少Times New Roman时会以Liberation Serif替代，PPT中保留Times New Roman指定。600 dpi PNG/TIFF由同一PDF生成。

本图与指定定稿目录中的Fig1–3、Fig5共享152窗口/66人的HI轴、外折及状态定义。未改动其他定稿图。S4编号对应用户指定附加图4；投稿前正文补充图引用应使用S4。

## 主要结论

锁定Ridge：R²=0.7286945328，MAE=0.4623687243。Persistence：R²=0.6622037933，MAE=0.5094831525。前次变化延续：R²=0.0006675206，MAE=0.9192004901。
加入4个MBX模块：R²=0.7328219679，MAE=0.4590849151；配对ΔMAE=−0.0032838092，95%受试者bootstrap区间[−0.0165131054,0.0094243301]，不支持明确改善结论。连续终点不能与Fig5事件AUROC直接比较。
校准斜率1.0151845367，95% CI [0.9454881146,1.0700054008]；截距−0.0188055687，95% CI [−0.0860732767,0.0531431249]。训练内变化量收缩系数均为1，因此校准前后预测相同，F/H及G/I相同是计算结果。

## 复算

Python需要numpy、pandas、scipy、scikit-learn、joblib。运行：
`python code/recompute.py`
`python code/verify.py`
图形构建使用@oai/artifact-tool与Codex presentation finalizer，需该运行环境及node_modules；build.mjs记录所有原生对象和坐标。其路径以本次工作目录为基础。

## 数据层次与解释边界

locked_predictions.csv和HI308_axis_fold_*为定稿HI中间数据，来自已审核的Fig1原始矩阵及分层建模管线。本任务不重新训练上游HI以免改变锁定定义。locked_HI_pipeline_reference.py记录该构建方法，作为出处，不是本资料包独立运行入口。
模块原始稀疏矩阵、特征表、样本元数据、100个上游模块模型及逐折得分均包含；verify.py从这些矩阵和模型复现模块输入，再复现10个下游模型。upstream_source_reference.py记录原Fig5训练方法，依赖原Fig5完整目录，是参考代码，不是独立入口。模块alpha在外折训练内选择，上游C保持定稿层的训练内选择。因此模块比较属于条件性的探索分析，不等于全管线重新嵌套验证。MAE区间是固定模型OOF预测的受试者抽样区间，不包括重训不确定性。

内折调参A是六种特征族13个alpha，共78组配置，每个外折训练内做4折分组预测；原有包含实际dt23的78组配置被删除。完整390个内折汇总MAE、六种族的嵌套外折预测、逐折选择、收缩输入和分箱成员均在data内。没有使用未来间隔预测，没有新增虚构实验，没有临床复发标签。

MANIFEST_SHA256.tsv给出最终包内每个文件的SHA256，以便识别版本。
