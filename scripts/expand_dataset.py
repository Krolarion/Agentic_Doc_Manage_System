# 扩展训练数据集：生成240篇文档 + 5000+ QA 对
import sys, fitz, os, json, uuid, time, random, hashlib, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
random.seed(2026)

DATA_DIR = "e:/Document_Management_System/Agentic_Doc_System/data"
TEST_DATA = "e:/Document_Management_System/Agentic_Doc_System/test_data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEST_DATA, exist_ok=True)

# ═══════════════════════════════════════
# 扩大的参数池
# ═══════════════════════════════════════
CO = [f"{c}科技有限公司" for c in [
    "华远", "中诚云", "天启智能", "锐思数据", "博远信息", "鼎新软件", "领航数字", "万维创新",
    "深蓝AI", "明源数据", "恒通电子", "金盾安全", "星辰微电子", "智联未来", "云帆数据",
    "极光网络", "经纬软件", "磐石信息", "蓝海数据", "创想空间",
]]
NAMES = ["张建国","李明远","王建华","陈志强","刘伟明","赵刚毅","孙涛","周文博",
         "陈晓东","杨帆","黄磊","许昌明","何平","吕良","施明哲","沈涛","韩冰","崔浩",
         "王芳","李工","张工","赵强","刘洋","马丁","胡伟","林涛",
         "陈思远","吴敏","郑洁","谢琳","唐亮","冯雪","周强","徐明"]
CITIES = ["北京","上海","深圳","杭州","成都","广州","武汉","南京","西安","苏州"]
DEPT = ["研发中心","技术部","产品部","市场部","财务部","人力资源部","运维部","法务部","销售部","质量部"]

# ═══════════════════════════════════════
# 12 类文档模板
# ═══════════════════════════════════════

def doc_contract():
    c1, c2 = random.sample(CO, 2)
    return f"""软件开发与技术服务合同

甲方（委托方）：{c1}
乙方（开发方）：{c2}

第一条 项目内容
甲方委托乙方开发{random.choice(['企业资源管理系统','客户关系管理平台','供应链协同系统','智能数据分析平台','自动化运维系统'])}，项目代号{random.choice(['Eagle','Phoenix','Atlas','Titan','Aurora'])}-{random.randint(100,999)}。

第二条 交付物
源代码及可执行程序、系统设计文档、数据库设计文档、接口规范文档、用户操作手册、系统部署与运维手册、测试报告含性能测试数据。

第三条 开发周期与里程碑
总工期{random.randint(6,18)}个月。需求分析完成：{random.randint(20,40)}个工作日；原型设计完成：{random.randint(15,30)}个工作日；Alpha版本交付：{random.randint(60,150)}个自然日；Beta版本交付及UAT测试：{random.randint(30,60)}个自然日；正式上线：全部工期届满前{random.randint(10,20)}个自然日。

第四条 合同价款与支付
合同总金额：人民币{random.randint(50,500)}万元整。签约后{random.randint(5,10)}个工作日内支付{random.randint(20,35)}%，Alpha交付后{random.randint(20,35)}%，正式验收后{random.randint(25,35)}%，质保期满后{random.randint(5,15)}%。

第五条 知识产权
项目成果的知识产权（含源代码著作权）归{random.choice(['甲方','双方共有'])}所有。乙方为本项目开发所使用的开源组件的许可证合规性由乙方负责，确保不侵犯第三方知识产权。

第六条 质保与维护
质保期{random.randint(12,36)}个月，免费修复缺陷和漏洞。质保期内系统可用性不低于{random.choice(['99.5%','99.9%'])}。质保期后维护费用：每年合同总金额的{random.randint(10,20)}%。

第七条 违约责任
乙方逾期交付：每逾期一日按合同总额万分之{random.randint(3,10)}支付违约金，逾期超过{random.randint(30,60)}日甲方可解除合同。交付成果严重不符合需求规格的，甲方可要求返工或解除合同。

第八条 保密
双方对合同内容及获知的对方商业信息永久保密。违反保密义务赔偿对方全部实际损失。

甲方签章：        乙方签章：
日期：202{random.randint(5,7)}.{random.randint(1,12)}.{random.randint(1,28)}
"""

def doc_whitepaper():
    co = random.choice(CO)
    year = random.randint(2024, 2026)
    topic = random.choice([
        "企业数字化转型方法论", "云原生架构最佳实践", "AI驱动的智能运维体系",
        "数据治理框架与实施路径", "零信任安全架构白皮书", "隐私计算技术及应用白皮书",
        "低代码平台技术选型指南", "实时流计算架构设计", "数字孪生技术在企业中的应用",
        "RPA流程自动化实践白皮书",
    ])
    return f"""{topic}

编制单位：{co}
版本：V{random.randint(1,3)}.{random.randint(0,9)}
发布日期：{year}年{random.randint(1,12)}月

摘要
本文档系统阐述{topic}的核心理念、技术架构与实施方法论。{co}结合在{random.choice(['金融','政务','制造','医疗','教育'])}行业的多年实践，总结出一套可落地的参考架构和实施路径。

1. 背景与挑战
传统IT架构面临运维成本高、资源利用率低、交付效率慢等挑战。根据行业调研，{random.randint(60,85)}%的企业正在或计划推进数字化转型，但其中{random.randint(30,55)}%的项目未能达到预期目标。核心痛点包括：数据孤岛严重、技术债务累积、组织协同困难。

2. 核心架构
推荐采用{random.choice(['分层微服务架构','事件驱动架构','数据湖仓一体架构'])}：基础设施层（容器化+K8s）、数据层（混合存储+缓存+搜索）、服务层（微服务+API网关+消息队列）、应用层（渐进式Web应用）。

3. 关键技术
{random.randint(5,10)}项关键技术：容器化（Docker/Podman）、服务网格（Istio/Linkerd）、声明式配置（GitOps）、可观测性（OpenTelemetry/Prometheus/Grafana/Tempo）、CI/CD流水线（{random.choice(['GitLab CI','Jenkins','ArgoCD'])}）、服务网格、混沌工程。

4. 实施路径
分{random.randint(3,5)}阶段推进：第一阶段（评估与规划，{random.randint(4,8)}周）完成现状评估、目标定义、路线图制定。第二阶段（试点验证，{random.randint(8,16)}周）选择{random.randint(2,4)}个代表性业务场景进行试点。第三阶段（规模推广）在全组织范围推广。

5. 度量与优化
建议建立以下度量体系：交付效率（部署频率、变更前置时间）、质量（变更失败率、平均恢复时间）、资源效率（云资源利用率、成本/TPS）、业务指标（功能上线周期、用户满意度）。

6. 案例分析
{co}为{random.choice(['某大型商业银行','某省级政务平台','某头部电商企业','某跨国制造集团','某三甲医院'])}成功实施了{topic}改造。项目周期{random.randint(8,18)}个月，投入{random.randint(200,1500)}万元。成果：系统部署频率从月级提升至日级，线上故障恢复时间从小时级降低至分钟级，运维人力成本降低{random.randint(30,55)}%，新功能上线周期缩短{random.randint(50,80)}%。

编制：{random.choice(NAMES)}（{random.choice(['架构师','技术总监','首席科学家'])}）
审核：{random.choice(NAMES)}
"""

def doc_audit():
    co = random.choice(CO)
    auditor = random.choice(["普华永道中天", "德勤华永", "安永华明", "毕马威华振", "立信", "天健", "信永中和"])
    year = random.randint(2023, 2026)
    return f"""信息安全审计报告

被审计单位：{co}
审计机构：{auditor}会计师事务所
审计期间：{year}年1月至{year}年12月
报告编号：IS-{random.randint(20240001,20269999)}

一、审计目标与范围
本次审计依据《信息安全技术网络安全等级保护基本要求》（等保{random.choice(['2.0','三级','二级'])}）、《ISO 27001信息安全管理体系标准》和行业监管要求。审计范围覆盖：网络安全管理体系、数据安全保护措施、应用系统安全、物理环境安全、外包与第三方管理、安全事件响应机制。

二、审计方法
文档审查、现场访谈、技术测试（漏洞扫描、渗透测试、配置核查）、日志分析、合规矩阵评估。

三、审计发现

3.1 高风险发现
审计发现高风险项{random.randint(1,4)}项：{random.choice(['部分核心系统存在未修复的高危漏洞','数据库访问控制策略过于宽松','生产环境与测试环境未有效隔离','敏感数据传输未全程加密'])}。{random.choice(['建议24小时内修复','已在审计期间完成整改','要求30天内制定整改方案并经审计确认'])}。

3.2 中风险发现
中风险项{random.randint(3,8)}项，主要包括：密码策略复杂度不够（最小长度{random.choice(['8','10'])}位要求未全面执行）、部分系统日志保存期限不足{random.choice(['6个月','1年'])}的要求、离职员工账号清理不及时、第三方供应商安全评估报告过期等。

3.3 低风险及改进建议
低风险项{random.randint(5,15)}项，涉及安全文档规范性、安全意识培训覆盖率、补丁管理流程优化等。

四、合规性评估
被审计单位在{random.randint(70,95)}%的审计检查项中满足合规要求。主要合规差距集中在：数据跨境传输管理、个人隐私保护机制、供应商安全评估。

五、审计结论
{random.choice(['出具保留意见，认为被审计单位已建立较完善的信息安全管理体系，但部分领域仍需持续改进','出具标准无保留意见，被审计单位信息安全管理体系运行有效','出具否定意见，建议被审计单位进行全面安全整改并重新评估'])}。

审计组长：{random.choice(NAMES)}（CISA/CISSP认证）
报告日期：{year+1}年{random.randint(1,6)}月{random.randint(1,30)}日
"""

def doc_training():
    co = random.choice(CO)
    course = random.choice(["新员工技术培训", "项目经理能力发展", "数据分析实战", "云计算认证培训", "AI算法进阶", "DevOps实践", "网络安全意识", "客户沟通技巧"])
    return f"""{co} {course}课程大纲

培训编号：TR-{random.randint(2024001,2026999)}
培训学时：{random.randint(16,40)}学时（{random.randint(2,5)}天）
培训对象：{random.choice(['新入职技术岗员工','项目经理','数据分析师','全栈开发工程师','各级管理者'])}
讲师：{random.choice(NAMES)}（{random.choice(['高级架构师','资深技术专家','特聘教授','认证培训师'])}）

一、培训目标
掌握{course}领域的核心知识体系与实操技能。通过考核的学员应具备独立承担相关工作的能力。

二、课程内容

模块一：基础概念与框架（{random.randint(3,6)}学时）
行业背景与发展趋势、核心术语与概念辨析、主流方法与框架。实操：搭建{random.choice(['开发环境','实验环境','数据分析平台'])}。

模块二：核心技术（{random.randint(6,12)}学时）
关键技术原理：{random.choice(['分布式系统一致性','机器学习模型训练','神经网络架构设计','数据仓库建模','微服务治理'])}。案例实战：{random.randint(3,6)}个hands-on实验。

模块三：进阶主题（{random.randint(4,8)}学时）
性能优化、安全加固、高可用设计、监控与告警。{random.choice(['生产事故案例分析','行业最佳实践分享','前沿技术趋势'])}。

模块四：综合考核（{random.randint(2,4)}学时）
理论笔试（{random.randint(60,120)}分钟）+ 实操考核（{random.randint(2,4)}小时）。

三、考核标准
总成绩 = 平时表现{random.randint(10,20)}% + 理论考试{random.randint(30,40)}% + 实操考核{random.randint(40,60)}%。总成绩{random.randint(70,80)}分以上为合格，{random.randint(85,90)}分以上为优秀。不合格者可免费参加下一期培训。

四、培训资源
配套教材、实验环境（每人独立{random.choice(['虚拟机','容器环境','云账号'])}）、预装软件的U盘。补充阅读材料：{random.randint(3,8)}篇推荐论文和{random.randint(2,5)}本参考书。

编制：{co}人力资源部
批准人：{random.choice(NAMES)}
"""

def doc_release():
    co = random.choice(CO)
    product = random.choice(["DocBrain", "DataVault", "CloudMesh", "AIMind", "SecuBot", "FlowWise"])
    ver = f"V{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"
    date = f"202{random.randint(5,7)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    return f"""{co} {product} {ver} 发布说明

发布日期：{date}

新增功能
- {random.choice(['智能问答引擎：支持基于RAG的多轮对话','实时数据管道：支持流式数据的低延迟处理','可视化工作流设计器：拖拽式编排','多租户管理：支持租户级别的资源隔离与配额管理','自然语言查询：用户可用自然语言描述需求自动生成SQL'])}
- {random.choice(['暗黑模式支持','移动端适配优化','国际化（中/英/日）','SSO单点登录集成LDAP/SAML/OAuth','自定义报表设计器'])}
- {random.choice(['AI辅助编码：代码补全和重构建议','一键部署到多云环境（AWS/Azure/阿里云）','审计日志支持导出为CSV/PDF','API速率限制和配额管理','实时协作编辑'])}
- {random.choice(['数据血缘追踪','异常检测告警','合规检查自动化','多语言支持','插件市场'])}。本次共新增{random.randint(8,25)}项功能

功能改进
- {random.choice(['查询性能优化：P95响应时间降低40%','UI重构：统一设计语言提升操作效率25%','安全加固：通过渗透测试修复12个漏洞','大数据量分页加载速度提升3倍','批量操作支持异步处理'])}
- {random.choice(['搜索精确度提升','权限模型细化','数据导出格式增加','错误提示友好化','日志保留期可配置'])}。本次共优化{random.randint(10,30)}项

废弃与移除
{random.choice(['旧版API V1将于6个月后停止服务，请迁移至V2','deprecated方法列表见API文档','从下个版本起不再支持IE11','旧版报表引擎将在3个月后移除'])}。

升级注意事项
{random.choice(['升级前请备份数据库','需先升级至上一个主要版本','Docker镜像大小增加约200MB','配置文件格式有变化，请参考迁移指南','本次升级需要停机约30分钟'])}。

贡献者：{', '.join(random.sample(NAMES, random.randint(3,8)))}
"""

def doc_report():
    co = random.choice(CO); p = random.choice(NAMES)
    d = f"202{random.randint(5,7)}.{random.randint(1,12)}.{random.randint(1,28)}"
    return f"""{co} {random.choice(['月度','季度','年度'])}项目进展报告

报告人：{p}（项目经理）
报告期间：{d}

一、项目概况
项目整体进度完成{random.randint(40,95)}%，当前处于{random.choice(['核心开发','集成测试','UAT','试运行'])}阶段，团队规模{random.randint(8,35)}人。

二、各模块进展
需求分析：{random.randint(90,100)}%完成。设计阶段：{random.randint(70,100)}%。开发阶段：{random.randint(30,85)}%完成。测试阶段：已编写{random.randint(50,300)}个测试用例，通过率{random.randint(85,98)}%。部署：{random.choice(['开发环境已搭建','测试环境就绪','预发布环境配置中'])}。

三、风险与问题
Top{random.randint(3,5)}风险：{random.choice(['关键模块技术难度超预期','第三方API服务不稳定','测试环境资源不足','需求变更频繁影响排期','核心开发人员离职需补充'])}。应对措施已在风险登记册中更新，由项目经理跟踪。

四、资源情况
预算使用率{random.randint(30,80)}%，人力投入累计{random.randint(200,1500)}人天。当前{random.choice(['有','无'])}资源缺口。

五、下阶段计划
重点任务：{random.choice(['完成核心模块联调','启动性能测试','组织UAT测试','准备上线材料','开展安全渗透测试'])}。预计下阶段完成{random.randint(15,35)}%的增量进度。

审批：{random.choice(NAMES)}（{random.choice(['部门总监','项目发起人','PMO负责人'])}）
"""

def doc_policy():
    co = random.choice(CO)
    dept = random.choice(["信息安全", "数据管理", "代码规范", "运维管理", "采购管理", "资产管理"])
    return f"""{co} {dept}管理制度

制定部门：{random.choice(DEPT)}
生效日期：202{random.randint(5,7)}年{random.randint(1,12)}月{random.randint(1,28)}日

第一章 总则
本制度适用于{co}及各下属子公司全体员工和外包人员。旨在规范{dept}管理，保障信息安全，提升运营效率。任何人违反本制度将视情节给予处分。

第二章 职责分工
{random.choice(['部门负责人负总责','信息安全委员会负责监督','审计部门负责定期检查'])}。各部门须设置{dept}管理员，负责日常执行。

第三章 管理要求
3.1 {random.choice(['所有系统须启用双因素认证','数据分级分为公开/内部/秘密/机密四级','代码提交须通过Code Review','生产环境变更须经变更管理委员会审批','采购超过5万元须三家比价'])}
3.2 {random.choice(['密码长度不少于12位含大小写数字和特殊字符','核心数据每日全量备份异地存储保留30天','依赖项须通过安全扫描后引入','监控告警须5分钟内响应','供应商须签署保密协议'])}
3.3 {random.choice(['每季度进行安全审计','每月进行漏洞扫描','每周进行代码质量审查','每日进行系统健康检查'])}。{random.choice(['违规操作将记入员工档案','重大违规将依法追究','累计违规三次解除劳动合同'])}。

第四章 检查与审计
内部审计部门每{random.choice(['季度','半年','年度'])}进行合规检查。检查结果向{random.choice(['管理层','董事会审计委员会','信息安全委员会'])}汇报。

第五章 培训与宣贯
新员工入职培训须含本制度章节，时长不少于{random.randint(1,4)}学时。全体员工每年参加一次{dept}管理培训，考试成绩{random.randint(70,80)}分以上合格。

第六章 附则
本制度由{random.choice(DEPT)}负责解释和修订。每{random.choice(['年度','两年'])}评审一次。

批准人：{random.choice(NAMES)}（{random.choice(['副总裁','CTO','CIO'])}）
"""

def doc_proposal():
    co = random.choice(CO)
    return f"""{co} {random.choice(['智能客服系统','数据中台','AI中台','微服务改造','安全加固'])}项目立项申请书

申请部门：{random.choice(DEPT)}
申请人：{random.choice(NAMES)}
申请日期：202{random.randint(5,7)}年{random.randint(1,12)}月{random.randint(1,28)}日

一、项目背景
当前{random.choice(['客服系统日均工单量已达{random.randint(500,5000)}单','数据分散在{random.randint(3,8)}个独立系统中','现有系统架构已运行{random.randint(3,8)}年','安全合规要求升级'])}。

二、项目目标
{random.choice(['工单自动处理率提升至{random.randint(50,80)}%','数据查询效率提升{random.randint(3,10)}倍','系统并发能力提升{random.randint(3,10)}倍','通过等保三级认证'])}。{random.choice(['ROI预计{random.randint(12,24)}个月内回收','年度运营成本降低{random.randint(20,45)}%','客户满意度提升{random.randint(15,30)}个百分点'])}。

三、技术方案
采用{random.choice(['微服务+容器化','云原生架构','混合云部署'])}。核心组件：{random.choice(['Spring Cloud + React','Go + Vue3','Python FastAPI + Next.js'])}。部署环境：{random.choice(['私有云','公有云','混合云'])}（{random.randint(5,20)}节点K8s集群）。

四、项目预算
总预算：{random.randint(100,2000)}万元。人力：{random.randint(5,20)}人×{random.randint(6,18)}个月。硬件/云资源：{random.randint(30,300)}万元。软件License：{random.randint(10,150)}万元。培训：{random.randint(5,50)}万元。预备金（{random.randint(10,20)}%）：{random.randint(10,200)}万元。

五、项目计划
{random.randint(6,18)}个月分{random.randint(3,5)}阶段完成。

六、预期收益
{random.choice(['年度节省人力成本{random.randint(50,500)}万元','新业务上线周期缩短{random.randint(50,80)}%','系统可用性从{random.choice([\"99%\",\"99.5%\"])}提升至99.9%','数据驱动决策覆盖率提升至{random.randint(70,95)}%'])}。

审批意见：    批准人：    日期：
"""


TEMPLATES = {
    "合同": doc_contract, "白皮书": doc_whitepaper, "审计报告": doc_audit,
    "培训": doc_training, "发布说明": doc_release, "项目报告": doc_report,
    "管理制度": doc_policy, "立项书": doc_proposal,
}
DOCS_PER_TYPE = 30  # 8类×30=240篇


# ═══════════════════════════════════════
# 生成 PDF
# ═══════════════════════════════════════

def write_pdf(text, path):
    doc = fitz.open()
    page = doc.new_page()
    y, x = 760, 56
    for para in text.split('\n'):
        para = para.strip()
        if not para:
            y -= 8; continue
        fs = 10.5 if len(para) < 50 and para[0] in '一二三四五六七八九十第123456' else 9.5
        for i in range(0, len(para), 55):
            if y < 45: page = doc.new_page(); y = 760
            page.insert_text((x, y), para[i:i+55], fontsize=fs, fontname='china-s')
            y -= fs * 1.65
    doc.save(path, deflate=True); doc.close()


# ═══════════════════════════════════════
# QA 生成
# ═══════════════════════════════════════

def generate_qa(filename, text, client):
    prompt = f"""为以下企业文档生成25-30条高质量问答测试对。

文档：{filename}
内容：{text[:5000]}

要求：问题多样化类型（事实查询/数字提取/定义解释/推理判断/综合理解）；难度覆盖easy/medium/hard；每条标注类型和难度；严格JSON数组格式。

格式：[{{"question":"...","answer":"...","difficulty":"easy|medium|hard","type":"事实查询|数字提取|定义解释|推理判断|综合理解"}}]"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role":"system","content":"你是专业测试数据标注专家，严格按JSON格式输出。"},{"role":"user","content":prompt}],
            temperature=0.2, response_format={"type":"json_object"},
        )
        raw = resp.choices[0].message.content
        if not raw: return []
        raw = re.sub(r'^```(?:json)?\s*\n?','',raw.strip()); raw = re.sub(r'\n?```\s*$','',raw.strip())
        result = json.loads(raw)
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list): return v
            return []
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"    QA ERROR: {e}"); return []


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def _save_dataset(path, qa_pairs, total_docs):
    dataset = {
        "name": "Enterprise QA Test Dataset",
        "version": "2.0",
        "created_at": datetime.now().isoformat(),
        "total_documents": total_docs,
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("扩展数据集：生成240篇文档 + 5000+ QA对")
    print("=" * 60)

    # 清理旧文档
    for f in os.listdir(DATA_DIR):
        if f.endswith('.pdf'): os.remove(os.path.join(DATA_DIR, f))

    # 1. 生成文档
    total_docs, total_chars = 0, 0
    for cat, fn in TEMPLATES.items():
        for i in range(DOCS_PER_TYPE):
            text = fn()
            fname = f"{cat}_{i+1:02d}.pdf"
            write_pdf(text, os.path.join(DATA_DIR, fname))
            total_docs += 1; total_chars += len(text)
        print(f"  [{cat}] {DOCS_PER_TYPE}篇 ({sum(1 for _ in [0])}字)")
    print(f"\n文档: {total_docs}篇, {total_chars:,}字, 平均{total_chars//total_docs}字/篇\n")

    # 2. 生成 QA（增量保存，断点续跑）
    from openai import OpenAI
    from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    save_path = os.path.join(TEST_DATA, "qa_test_dataset.json")
    all_qa = []
    # 如果已有部分数据，加载续跑
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                all_qa = old.get("qa_pairs", [])
                done_docs = set(q["source_doc"] for q in all_qa)
                print(f"  [续跑] 已有 {len(all_qa)} 条QA，已处理 {len(done_docs)} 篇文档")
        except: pass
    else:
        done_docs = set()

    pdfs = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.pdf'))
    for i, fname in enumerate(pdfs):
        doc = fitz.open(os.path.join(DATA_DIR, fname))
        text = ''.join(p.get_text('text') for p in doc); doc.close()
        qa_list = generate_qa(fname, text, client)
        cat = fname.split('_')[0]
        for qa in qa_list:
            qa["qa_id"] = f"qa_{uuid.uuid4().hex[:8]}"
            qa["source_doc"] = fname; qa["category"] = cat
        all_qa.extend(qa_list)

        # 每20篇保存一次
        if (i+1) % 20 == 0:
            _save_dataset(save_path, all_qa, total_docs)
            print(f"  [{i+1}/{len(pdfs)}] {len(all_qa)} QA (已保存)")

        time.sleep(0.3)

    # 最终保存
    _save_dataset(save_path, all_qa, total_docs)
    print(f"\n  [DONE] {len(all_qa)} QA 已保存到 {save_path}")

    # 统计
    cats, diffs, types = {}, {}, {}
    for q in all_qa:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
        diffs[q.get("difficulty", "?")] = diffs.get(q.get("difficulty", "?"), 0) + 1
        types[q.get("type", "?")] = types.get(q.get("type", "?"), 0) + 1

    print(f"\nQA数据集统计:")
    print(f"  总QA对: {len(all_qa)}")
    print(f"  分类: {cats}")
    print(f"  难度: {diffs}")
    print(f"  类型: {types}")
    print(f"  已保存: {save_path}")


if __name__ == "__main__":
    main()
