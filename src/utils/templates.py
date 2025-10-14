"""
Templates for the LLM
"""

import textwrap


class Templates:
    """
    Templates for the LLM
    """

    def __init__(self):
        # set role and task
        self._purellm_sys_msg_1 = self._strip_whitespace("""\
            角色：你是资深金融风控建模专家，熟悉信用评分、逾期定义、卡账行为、稳定性检验。
            
            任务：基于给定用户的个人信息和信用卡历史数据，预测其在最新一个账单周期是否能按时还款。只使用提供的数据，不要臆测缺失值。
        """)

        # output format
        self._purellm_sys_msg_2 = self._strip_whitespace("""\
            请输出合法JSON格式（不包含 markdown code block），包含以下字段：
            {
                "is_delinquent": boolean
            }
        """)
        self._purellm_sys_msg_3 = self._strip_whitespace("""\
            请输出合法JSON格式（不包含 markdown code block），包含以下字段:
            {
                "reasoning_process": str,
                "delinquency_prob": float,
                "risk_level": "low|medium|high",
                "is_delinquent": boolean
            }
        """)

        # user message
        self._purellm_user_msg_1 = self._strip_whitespace("""\
            【用户信息】：
            - 居住地：南京
            - 所属分支行为：{lvl_4_bch_nam}
            - 居住情况：{residence}
            - 行业：{industry}
            - 学历：{education}
        """)
        self._purellm_user_msg_2 = self._strip_whitespace("""\
            - 出生年份：{birth_year}
            - 性别：{sex}
            - 婚姻状态：{marriage_status}
        """)
        self._purellm_user_msg_3 = self._strip_whitespace("""\
            【消费与历史违约情况】：
            你过去的消费与违约情况如下：
            {transaction_history}
        """)

    def _strip_whitespace(self, text: str):
        return textwrap.dedent(text).strip()

    def get_purellm_sys_msg(self, is_cot_prompt: bool):
        if is_cot_prompt:
            return f"{self._purellm_sys_msg_1}\n\n{self._purellm_sys_msg_3}"
        if not is_cot_prompt:
            return f"{self._purellm_sys_msg_1}\n\n{self._purellm_sys_msg_2}"

    def get_purellm_user_msg(self, has_protected_attributes: bool):
        if has_protected_attributes:
            return f"{self._purellm_user_msg_1}\n{self._purellm_user_msg_2}\n\n{self._purellm_user_msg_3}"
        if not has_protected_attributes:
            return f"{self._purellm_user_msg_1}\n\n{self._purellm_user_msg_3}"


templates = Templates()
