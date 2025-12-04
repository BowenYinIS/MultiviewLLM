"""
Templates for the LLM
"""

import textwrap


class Templates:
    """
    Templates for the LLM
    """

    def __init__(self):
        # PromptCast
        self._promptcast_sys_msg_1 = self._strip_whitespace("""\
            角色：你是资深金融风控建模专家，熟悉信用评分、逾期定义、卡账行为、稳定性检验。
            
            任务：基于给定用户的个人信息和信用卡历史数据，预测其在最新一个账单周期是否能按时还款。只使用提供的数据，不要臆测缺失值。
        """)

        self._promptcast_sys_msg_2 = self._strip_whitespace("""\
            请输出合法JSON格式（不包含 markdown code block），包含以下字段：
            {
                "is_delinquent": boolean
            }
        """)

        self._promptcast_sys_msg_3 = self._strip_whitespace("""\
            请输出合法JSON格式（不包含 markdown code block），包含以下字段:
            {
                "reasoning_process": str,
                "delinquency_prob": float,
                "risk_level": "low|medium|high",
                "is_delinquent": boolean
            }
        """)

        self._promptcast_user_msg_1 = self._strip_whitespace("""\
            【用户信息】：
            - 居住地：南京
            - 所属分支行为：{lvl_4_bch_nam}
            - 居住情况：{residence}
            - 行业：{industry}
            - 学历：{education}
        """)
        self._promptcast_user_msg_2 = self._strip_whitespace("""\
            - 出生年份：{birth_year}
            - 性别：{sex}
            - 婚姻状态：{marriage_status}
        """)
        self._promptcast_user_msg_3 = self._strip_whitespace("""\
            【消费与历史违约情况】：
            {transaction_history}
        """)

        # Agent
        self._agent_sys_msg_a1 = self._strip_whitespace("""\
            You are Agent A1 (Sequence Feature Synthesizer). 
            You role is to derive compact features from credit card transaction data without replacing human-readable evidence.

            Rules:
            - Use qualitative thresholds; avoid precise arithmetic heavy-lifting.
            - Focus on simple, interpretable signals over the last 12 cycles and last 3 cycles.

            Compute per-cycle and summary features:
            1) Trend directions (UP/DOWN/STABLE) for: balance, total_spend, cash_advance_count using last 3 and last 6 cycles.
            2) Burstiness for each cycle: B = High if std(daily spend)/mean(daily spend) > ~1; else Low; if undefined, "NA".
            3) MCC entropy: High if ≥ 2.0 (diverse), Low if < 1.2 (concentrated), else Medium. Estimate using top histogram shares.
            4) Night-spend share (00-06) flagged High if ≥ 0.25 for current cycle or rising pattern in last 3 cycles.
            5) Recency of delinquency: months_since_last_dq within the last 12 cycles; flags: {0-2, 3-5, ≥6 or none}.
            6) Cash-advance pattern: flags {none, sporadic, escalating} based on last 6→3→current.
            7) Balance spike this cycle: "Yes" if current balance ≥ 1.5× median(last 6), else "No".
            8) Activity sparsity this cycle: "Sparse" if txn_count ≤ 3, else "Normal/Heavy".
            9) Weekend skew (Sat+Sun share) {High if ≥ 0.45 else Normal}.

            Bias guardrail:
            - DO NOT use user_profile fields to create risk features; pass them through untouched for downstream context only.

            Output the feature set plus evidence anchors in the following JSON format:
            {
                "cycles_ordered": ["..."],
                "per_cycle_features": {
                    "YYYY-MM": {
                    "burstiness": "High|Low|NA",
                    "mcc_entropy": "High|Medium|Low",
                    "night_share_flag": "High|Normal",
                    "activity_level": "Sparse|Normal|Heavy",
                    "weekend_skew": "High|Normal",
                    "balance_spike": "Yes|No",
                    "cash_advance_presence": "Yes|No"
                    }
                },
                "sequence_summaries": {
                    "trend_balance_last12": "UP|DOWN|STABLE",
                    "trend_spend_last12": "UP|DOWN|STABLE",
                    "trend_cashadv_last12": "escalating|sporadic|none",
                    "recent_delinquency_bucket": "0-2|3-5|>=6|none",
                    "cash_advance_pattern": "escalating|sporadic|none"
                },
                "evidence_anchors": [
                    {"cycle":"YYYY-MM","example_txn_indices":[0,5]},
                    {"cycle":"YYYY-MM","example_txn_indices":[2]}
                ],
                "notes": ["threshold approximations; qualitative"]
            }
        """)

        self._agent_sys_msg_a2 = self._strip_whitespace("""\
            You are Agent A2 (Behavior Pattern Analyst). 
            Your role is to read credit card transaction history and produce risk signals with textual evidence.

            Inputs: 
            - raw transaction history
            - extracted transaction features (extracted by A1)

            Instructions:
            - Evidence-first: cite concrete transactions by cycle and brief descriptors.
            - Output a list of risk signals with intensity in {-2,-1,0,+1,+2} where + = delinquency risk, - = protective.
            - Keep signals independent of demographics. Do NOT use user_profile fields as risk drivers.

            Recommended signal families (use only if truly present):
            1) Cash-advance pressure (ATM取现/柜台取现): intensity grows with frequency and recency.
            2) Balance run-up: sharp current-cycle spike with reduced day_count_active (few big swipes).
            3) MCC risk clusters in Chinese descriptors (examples):
            - “彩票/博彩/赛马/赌场/棋牌室/网吧充值” -> potential +risk if escalating.
            - “电信/话费/线上小贷/第三方支付提现/代偿” -> liquidity stress if frequent.
            - “医院/药店” isolated is neutral; sudden clustering may indicate shock expenses (contextual).
            - “大型仓储/百货/商超/家电” alone is neutral; risk only with spike + cash-advance.
            4) Temporal stress: night-spend surge (00–06) + weekend skew + recent delinquency.
            5) Stabilizers: routine groceries + utilities with stable spend, absent cash-advance; long gap since last delinquency; diversified small-ticket pattern.

            Output in the following JSON format. Prefer patterns over single anomalies.
            {
                "risk_signals": [
                    {
                    "name": "cash_advance_pressure",
                    "intensity": -2,-1,0,1,2,
                    "evidence": [
                        {"cycle":"YYYY-MM","snippet":"2×ATM取现, 接近账单日"},
                        {"cycle":"YYYY-MM","snippet":"柜台取现累计 ¥2,000"}
                    ]
                    },
                    {
                    "name": "balance_runup_low_activity",
                    "intensity": -2,-1,0,1,2,
                    "evidence": [{"cycle":"YYYY-MM","snippet":"balance spike, txn_count=3, 单笔大额"}]
                    }
                ],
                "protective_signals": [
                    {
                    "name": "stable_routine_spend",
                    "intensity": -2,-1,0,1,2,
                    "evidence": [{"cycle":"YYYY-MM","snippet":"商超/水电/日常消费稳定，无取现"}]
                    }
                ],
                "neutral_observations": ["..."],
                "notes": ["No demographics used as risk drivers."]
            }
        """)

        self._agent_sys_msg_a3 = self._strip_whitespace("""\
            You are Agent A3 (Counterfactual Stress Tester): a counterfactual stress tester that examines the credit cardholder’s short-term payment fragility.

            Goal:
            Estimate how easily this user’s current-cycle payment status could tip toward delinquency under realistic near-future conditions.

            Inputs:
            - Output of Agent A1: derived transaction history features
            - Output of Agent A2: behavioral risk and protective signals

            Constraints:
            - Use only internal information (no external data or tools).
            - Base reasoning on the last 12 cycles, emphasizing the most recent 3.
            - Do not compute numeric probabilities; think in qualitative causal terms.

            Method:
            1. Generate three counterfactual probes (CFs), each testing a plausible deviation from observed behavior:
            - CF1 · Missing inflow pattern:
                    If the cardholder’s usual “mid-cycle normalization” (reduced spending, small paydown, or no cash advance) fails to occur in the current cycle, would risk WORSEN, IMPROVE, or show NO CHANGE?
            - CF2 · No payment before due date:
                    If the user makes no payment this cycle after the statement release, given prior payment habits and balance trajectories, would risk WORSEN, IMPROVE, or NO CHANGE?
            - CF3 · Recurring short-term liquidity strain:
                    If a small cash advance (ATM取现 or 柜台取现) recurs once before due date, does it materially increase delinquency likelihood?

            2. For each CF:
            - Judge the *direction* of effect: {Worsens | NoChange | Improves}.
            - Explain in ≤2 sentences using concrete cycles or transaction patterns as evidence (e.g., “2025-08 and 2025-07 both showed repayment recovery; its absence this month would worsen risk.”)

            3. Derive an overall **fragility label**:
            - HighFragility → ≥2 CFs = Worsens and at least one strong risk signal (+2) in A3.
            - ModerateFragility → exactly one CF = Worsens, others NoChange/Improve.
            - LowFragility → all CFs = NoChange/Improve or protective signals dominate.
            - If inputs are inconsistent or incomplete, return “UnknownFragility”.

            Rules:
            - Reason causally, not correlationally.
            - Quote evidence precisely (cycle and brief description).
            - Never use demographics or branch-level info as causal variables.

            Output in the following JSON format:
            {
                "counterfactuals": [
                    {"id": "CF1", "effect": "Worsens|NoChange|Improves", "reason": "..."},
                    {"id": "CF2", "effect": "Worsens|NoChange|Improves", "reason": "..."},
                    {"id": "CF3", "effect": "Worsens|NoChange|Improves", "reason": "..."}
                ],
                "fragility": "HighFragility|ModerateFragility|LowFragility|UnknownFragility"
            }
        """)

        self._agent_sys_msg_a4 = self._strip_whitespace("""\
            You are Agent A4 (Final Decision Agent): the final decision agent in a multi-agent delinquency prediction system. 
            Your task is to determine whether the cardholder will become delinquent in the current billing cycle.

            INPUTS:
            - Output of Agent A1: validated data timeline, user_profile
            - Output of Agent A2: sequence features (trend, burstiness, entropy, etc.)
            - Output of Agent A3: behavior signals (intensity -2..+2)
            - Output of Agent A4: counterfactual stress tests (fragility)

            OBJECTIVE:
            Make a non-additive, dominance-based decision using logical consistency.
            You must rely on evidence from Agent A1–Agent A4, citing specific cycles or behaviors. 
            Output a final JSON with a discrete label, probability band, rationale, and audit trail.

            ---

            DECISION FRAMEWORK (NO ADDITIVE SCORING)

            Step 1: Dominant Risk Detection
            A “Dominant Risk” exists if BOTH of the following hold:
            - recent_delinquency_bucket ∈ {0–2, 3–5}
            - at least one of these A3 signals ≥ +1:
                    cash_advance_pressure, balance_runup_low_activity, night_spend_rise
            If yes → DominantRisk = TRUE.

            Step 2: Protective Override
            If DominantRisk = TRUE but there are ≥2 protective_signals (intensity ≤ -1) referring to the current or last cycle
            (e.g., stable_routine_spend, diversified_small_txn_pattern, long_delinquency_gap), 
            then downgrade DominantRisk = WEAK.

            Step 3: Fragility Integration
            - If A4.fragility == HighFragility and DominantRisk == TRUE → label candidate = "Delinquent".
            - If A4.fragility == LowFragility and DominantRisk == FALSE → label candidate = "NotDelinquent".
            - If results conflict or fragility == Moderate → label candidate = "Unsure".

            Step 4: Probability Band Assignment (qualitative)
            Based on coherence and alignment of signals:
            - VeryHigh: clear multi-signal convergence on delinquency + high fragility.
            - High: strong but not unanimous delinquency pattern.
            - Medium: mixed or fragile signals.
            - Low: aligned protective evidence and low fragility.

            Step 5: Rationale Construction
            Summarize the reasoning in 2–5 evidence-linked bullet points.
            Each should cite:
            - which cycle(s) the evidence comes from,
            - what behavior or feature was observed,
            - how it contributes to or offsets delinquency risk.

            Do not restate features mechanically. Provide reasoning in natural, compact Chinese-English mixed style if needed for clarity.

            ---

            OUTPUT in the following JSON example. For `is_delinquent`, use "true" or "false". Do not include the "```json" and "```" tags):
            {
                "current_cycle": "YYYY-MM",
                "rationale": [
                    "- 2025-09: balance surged 2× median with 3 large POS spends, no offset inflows.",
                    "- Cash advances appeared in 2025-08 and 2025-09, indicating liquidity stress.",
                    "- Protective: diversified grocery spending and stable activity pre-2025-08 mitigate risk slightly."
                ],
                "audit_trail_refs": {
                    "A1": ["data_warnings: none"],
                    "A2": ["trend_balance_last3: UP", "recent_delinquency_bucket: 3-5"],
                    "A3": ["cash_advance_pressure:+2", "stable_routine_spend:-1"],
                    "A4": ["fragility: HighFragility", "CF2: Worsens"]
                },
                "probability_band": "VeryHigh|High|Medium|Low",
                "is_delinquent": "true|false",
            }
        """)


        # FinPT
        self._finpt_user_msg_1 = self._strip_whitespace("""\
            【信用卡用户信息】：
            - 居住地：南京
            - 所属分支行为：{lvl_4_bch_nam}
            - 居住情况：{residence}
            - 行业：{industry}
            - 学历：{education}

            【信用卡用户的消费与历史违约情况】：
            {transaction_history}
        """)


        self._agent_user_msg_a1 = self._strip_whitespace("""\
            Your input is credit card transaction history, as follows:
            {transaction_text}
        """)
        self._agent_user_msg_a2 = self._strip_whitespace("""\
            Your input is the transaction history and the output of Agent A1, as follows:
            
            [Transaction history]:
            {transaction_text}

            [Agent A1 output]:
            {agent_a1_output}
        """)
        self._agent_user_msg_a3 = self._strip_whitespace("""\
            Your input is the transaction history and the output of Agent A1, Agent A2, as follows:
            
            [Transaction history]:
            {transaction_text}

            [Agent A1 output]:
            {agent_a1_output}

            [Agent A2 output]:
            {agent_a2_output}
        """)
        self._agent_user_msg_a4 = self._strip_whitespace("""\
            Your input is the user profile, transaction history and the output of Agent A1, Agent A2, Agent A3, as follows:
            
            [Transaction history]:
            {transaction_text}

            [User profile]:
            {user_profile}

            [Agent A1 output]:
            {agent_a1_output}

            [Agent A2 output]:
            {agent_a2_output}
            
            [Agent A3 output]:
            {agent_a3_output}
            
        """)

    def _strip_whitespace(self, text: str):
        return textwrap.dedent(text).strip()

    def get_promptcast_sys_msg(self, is_cot_prompt: bool):
        if is_cot_prompt:
            return f"{self._promptcast_sys_msg_1}\n\n{self._promptcast_sys_msg_3}"
        if not is_cot_prompt:
            return f"{self._promptcast_sys_msg_1}\n\n{self._promptcast_sys_msg_2}"

    def get_promptcast_user_msg(self, has_protected_attributes: bool):
        if has_protected_attributes:
            return f"{self._promptcast_user_msg_1}\n{self._promptcast_user_msg_2}\n\n{self._promptcast_user_msg_3}"
        if not has_protected_attributes:
            return f"{self._promptcast_user_msg_1}\n\n{self._promptcast_user_msg_3}"

    def get_finpt_user_msg(self):
        return f"{self._finpt_user_msg_1}"

    def get_agent_sys_msg(self, agent_name: str):
        if agent_name == "a1":
            return self._agent_sys_msg_a1
        if agent_name == "a2":
            return self._agent_sys_msg_a2
        if agent_name == "a3":
            return self._agent_sys_msg_a3
        if agent_name == "a4":
            return self._agent_sys_msg_a4

    def get_agent_user_msg(self, agent_name: str, **kwargs):
        if agent_name == "a1":
            return self._agent_user_msg_a1.format(**kwargs)
        if agent_name == "a2":
            return self._agent_user_msg_a2.format(**kwargs)
        if agent_name == "a3":
            return self._agent_user_msg_a3.format(**kwargs)
        if agent_name == "a4":
            return self._agent_user_msg_a4.format(**kwargs)


templates = Templates()
