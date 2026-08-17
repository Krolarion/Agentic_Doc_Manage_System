# 进化策略Prompt优化器 — 基于CMA-ES思想的离散文本优化
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from src.agent.agent import DocumentAgent
from src.retrieval.search_engine import HybridSearchEngine
from src.storage.database_manager import DatabaseManager
from src.evaluation.judge import LLMJudge

# 候选prompt原子（每一条都可能加入system prompt）
PROMPT_ATOMS = [
    # 知识边界（核心）
    {"text": "如果你在知识库中找不到相关信息，必须诚实地说'知识库中未找到相关内容'，绝对不要编造。", "group": "boundary"},
    {"text": "你的所有回答必须严格基于检索到的文档内容，不要使用你自己的知识。", "group": "boundary"},
    {"text": "当检索结果不充分时，优先承认信息不足，而不是给出不完整的推测。", "group": "boundary"},
    {"text": "如果检索结果只覆盖了问题的部分内容，明确告知用户哪些部分有答案，哪些没有。", "group": "boundary"},

    # 检索策略
    {"text": "如果第一次检索结果不理想，换用不同关键词再检索一次，而不是直接编造。", "group": "search"},
    {"text": "优先使用精确的关键词进行检索，避免过于宽泛的查询。", "group": "search"},
    {"text": "检索后，先检查结果是否与问题相关，再决定是否回答。", "group": "search"},
    {"text": "对于需要多步推理的问题，先检索核心概念，再根据初步结果进行补充检索。", "group": "search"},

    # 引用规范
    {"text": "每条事实陈述后，用 [来源: 文件名] 标注出处。", "group": "citation"},
    {"text": "回答末尾列出所有引用的文档。", "group": "citation"},
    {"text": "如果答案来自多个文档片段，分别标注每个部分的来源。", "group": "citation"},

    # 回答质量
    {"text": "用简洁清晰的语言回答，避免不必要的背景介绍。", "group": "quality"},
    {"text": "对于专业术语，在回答中给出简短解释。", "group": "quality"},
    {"text": "把最重要的信息放在回答的开头。", "group": "quality"},
    {"text": "用 bullet points 组织多条信息，使回答更易读。", "group": "quality"},
]

# 基础prompt（不可变部分）
BASE_PROMPT = """你是一个企业文档智能助手，可以帮助用户在文档知识库中查找信息、回答问题。

## 你的能力
- 检索知识库中的文档内容（search_knowledge）
- 查看文档详情和元数据（get_document）
- 浏览知识库中的所有文档（list_documents）
- 获取知识库使用统计（get_stats）
- 阅读完整chunk内容（get_chunk）"""


@dataclass
class Individual:
    """种群个体：一个prompt方案"""
    genes: np.ndarray               # 0/1数组，表示每个atom是否启用
    fitness: float = 0.0            # 适应度（平均F%）
    avg_faithfulness: float = 0.0
    avg_accuracy: float = 0.0
    avg_relevance: float = 0.0

    def build_prompt(self) -> str:
        """将基因组装为完整prompt"""
        parts = [BASE_PROMPT]
        for i, atom in enumerate(PROMPT_ATOMS):
            if self.genes[i] == 1:
                parts.append(f"- {atom['text']}")
        return "\n".join(parts)


class PromptOptimizer:
    """
    进化策略Prompt优化器。
    用GA (遗传算法) + CMA-ES风格的自适应变异率，
    搜索最优prompt组合以最大化检索回答的忠实度 (F%)。
    """

    def __init__(self, db: DatabaseManager, engine: HybridSearchEngine,
                 pop_size: int = 20, generations: int = 12, elite_ratio: float = 0.3):
        self.db = db
        self.engine = engine
        self.judge = LLMJudge()
        self.pop_size = pop_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.n_atoms = len(PROMPT_ATOMS)
        self.population: List[Individual] = []
        self.best: Optional[Individual] = None
        self.history: List[Dict] = []

        # CMA-ES风格：自适应变异率
        self.mutation_rate = 0.15
        self.momentum = 0.9
        self.prev_improvement = 0.0

    def _create_agent(self, prompt: str) -> DocumentAgent:
        """创建使用指定prompt的Agent"""
        agent = DocumentAgent()
        agent.messages = [{"role": "system", "content": prompt}]
        return agent

    def _evaluate_individual(self, ind: Individual, test_queries: List[str]) -> Individual:
        """评估一个prompt方案"""
        prompt = ind.build_prompt()
        agent = self._create_agent(prompt)
        scores = []

        for q in test_queries:
            # 检索
            results = self.engine.search(q, top_k=5)
            contexts = "\n\n".join(r.content for r in results) if results else "(无)"

            # 生成
            agent.reset()
            agent.messages = [{"role": "system", "content": prompt}]
            answer = agent.chat(q, verbose=False)

            # 评分
            score = self.judge.evaluate(q, answer, contexts)
            scores.append(score)

        ind.fitness = sum(s.faithfulness for s in scores) / len(scores) if scores else 0
        ind.avg_faithfulness = ind.fitness
        ind.avg_accuracy = sum(s.accuracy for s in scores) / len(scores) if scores else 0
        ind.avg_relevance = sum(s.relevance for s in scores) / len(scores) if scores else 0
        return ind

    def _initialize_population(self):
        """初始化种群：随机个体 + 种子个体"""
        self.population = []

        # 种子个体：全部开启（当前默认行为）
        seed = Individual(genes=np.ones(self.n_atoms, dtype=int))
        self.population.append(seed)

        # 种子2：只开启boundary组
        boundary_only = np.zeros(self.n_atoms, dtype=int)
        for i, atom in enumerate(PROMPT_ATOMS):
            if atom["group"] == "boundary":
                boundary_only[i] = 1
        self.population.append(Individual(genes=boundary_only))

        # 随机个体
        for _ in range(self.pop_size - len(self.population)):
            genes = np.random.randint(0, 2, self.n_atoms)
            # 确保至少开启一个boundary类的atom
            boundary_idx = [i for i, a in enumerate(PROMPT_ATOMS) if a["group"] == "boundary"]
            if not any(genes[i] == 1 for i in boundary_idx):
                genes[random.choice(boundary_idx)] = 1
            self.population.append(Individual(genes=genes))

    def _select(self) -> List[Individual]:
        """锦标赛选择"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        n_elite = max(2, int(self.pop_size * self.elite_ratio))
        return sorted_pop[:n_elite]

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """均匀交叉"""
        mask = np.random.randint(0, 2, self.n_atoms)
        child_genes = np.where(mask, parent1.genes, parent2.genes)
        return Individual(genes=child_genes)

    def _mutate(self, ind: Individual):
        """CMA-ES风格自适应变异：根据最近改进动态调整变异率"""
        flip_mask = np.random.random(self.n_atoms) < self.mutation_rate
        ind.genes = np.where(flip_mask, 1 - ind.genes, ind.genes)

        # 保护：至少保留一个boundary原子
        boundary_idx = [i for i, a in enumerate(PROMPT_ATOMS) if a["group"] == "boundary"]
        if not any(ind.genes[i] == 1 for i in boundary_idx):
            ind.genes[random.choice(boundary_idx)] = 1

    def _update_mutation_rate(self, improvement: float):
        """自适应变异率：进展慢 → 加大变异；进展快 → 减小变异"""
        smoothed = self.momentum * self.prev_improvement + (1 - self.momentum) * improvement
        if smoothed < 1.0:  # 改进不明显
            self.mutation_rate = min(0.35, self.mutation_rate * 1.2)
        else:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
        self.prev_improvement = smoothed

    def run(self, test_queries: List[str], verbose: bool = True) -> Individual:
        """运行进化优化"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Prompt 进化优化 (CMA-ES style GA)")
            print(f"种群: {self.pop_size} | 代数: {self.generations} | 原子: {self.n_atoms}")
            print(f"测试集: {len(test_queries)} 条")
            print(f"{'='*60}")

        self._initialize_population()

        # 评估初始种群
        if verbose:
            print(f"\n[Gen 0] 评估初始种群...")
        for i, ind in enumerate(self.population):
            self._evaluate_individual(ind, test_queries)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.population[0]
        prev_best = self.best.fitness

        if verbose:
            print(f"  Best F%: {self.best.fitness:.1f}% | Active atoms: {self.best.genes.sum()}")

        # 进化循环
        for gen in range(1, self.generations + 1):
            elites = self._select()
            new_pop = list(elites)  # 精英保留

            # 交叉 + 变异生成子代
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(elites, min(2, len(elites)))
                child = self._crossover(p1, p2) if len(elites) >= 2 else Individual(genes=p1.genes.copy())
                self._mutate(child)
                new_pop.append(child)

            self.population = new_pop

            # 评估新个体（精英已评估过，跳过）
            for ind in self.population[len(elites):]:
                self._evaluate_individual(ind, test_queries)

            self.population.sort(key=lambda x: x.fitness, reverse=True)
            current_best = self.population[0]
            improvement = current_best.fitness - prev_best
            self._update_mutation_rate(improvement)

            if current_best.fitness > self.best.fitness:
                self.best = current_best

            prev_best = current_best.fitness

            if verbose:
                avg_fit = sum(ind.fitness for ind in self.population) / len(self.population)
                print(f"  [Gen {gen}] Best F%: {current_best.fitness:.1f}% | "
                      f"Pop avg: {avg_fit:.1f}% | Mut: {self.mutation_rate:.3f} | "
                      f"Atoms: {current_best.genes.sum()}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"优化完成!")
            print(f"  最佳 F%: {self.best.fitness:.1f}%")
            print(f"  最佳 A%: {self.best.avg_accuracy:.1f}%")
            print(f"  最佳 R%: {self.best.avg_relevance:.1f}%")
            self._print_best_atoms()

        return self.best

    def _print_best_atoms(self):
        """打印最优方案启用的原子"""
        print(f"\n启用原子 ({int(self.best.genes.sum())}/{self.n_atoms}):")
        for i, atom in enumerate(PROMPT_ATOMS):
            status = "✅" if self.best.genes[i] else "❌"
            print(f"  {status} [{atom['group']}] {atom['text'][:70]}...")

    def get_optimized_prompt(self) -> str:
        """返回最优prompt"""
        if self.best:
            return self.best.build_prompt()
        return BASE_PROMPT

    def compare(self, test_queries: List[str]) -> Dict:
        """对比优化前后"""
        from src.agent.tools import init_tools
        init_tools(self.db, self.engine)

        # 原始prompt
        orig_agent = DocumentAgent()
        orig_scores = []
        for q in test_queries:
            results = self.engine.search(q, top_k=5)
            ctx = "\n\n".join(r.content for r in results) if results else "(无)"
            orig_agent.reset()
            ans = orig_agent.chat(q, verbose=False)
            orig_scores.append(self.judge.evaluate(q, ans, ctx))

        # 优化后prompt
        opt_prompt = self.get_optimized_prompt()
        opt_agent = self._create_agent(opt_prompt)
        opt_scores = []
        for q in test_queries:
            results = self.engine.search(q, top_k=5)
            ctx = "\n\n".join(r.content for r in results) if results else "(无)"
            opt_agent.reset()
            opt_agent.messages = [{"role": "system", "content": opt_prompt}]
            ans = opt_agent.chat(q, verbose=False)
            opt_scores.append(self.judge.evaluate(q, ans, ctx))

        return {
            "original": {
                "F": f"{sum(s.faithfulness for s in orig_scores)/len(orig_scores):.1f}%",
                "A": f"{sum(s.accuracy for s in orig_scores)/len(orig_scores):.1f}%",
                "R": f"{sum(s.relevance for s in orig_scores)/len(orig_scores):.1f}%",
            },
            "optimized": {
                "F": f"{sum(s.faithfulness for s in opt_scores)/len(opt_scores):.1f}%",
                "A": f"{sum(s.accuracy for s in opt_scores)/len(opt_scores):.1f}%",
                "R": f"{sum(s.relevance for s in opt_scores)/len(opt_scores):.1f}%",
            },
            "best_prompt": opt_prompt,
        }
