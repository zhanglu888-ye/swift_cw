from swift.rewards import ORM, orms
import json
import re

class FormatRewardFunction(ORM):
    def __call__(self, completions, **kwargs):
        """
        格式奖励：判断是否输出合法JSON且包含关键字段
        """
        rewards =[]
        for comp in completions:
            content = comp[0]['content'] if isinstance(comp, list) else comp
            
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content.strip()

            try:
                data = json.loads(json_str)
                if "event_analyses" in data and isinstance(data["event_analyses"], list):
                    if len(data["event_analyses"]) > 0 and "comprehensive_assessment" in data["event_analyses"][0]:
                        rewards.append(0.5)
                    else:
                        rewards.append(0.2)
                else:
                    rewards.append(0.0)
            except json.JSONDecodeError:
                rewards.append(-0.5)
        return rewards

class AccuracyRewardFunction(ORM):
    def __call__(self, completions, **kwargs):
        """
        准确度奖励：比对生成的分类与 ground truth
        """
        # 注意：在类方法中，solution 会通过 kwargs 传进来
        solution = kwargs.get('solution',[])
        
        rewards =[]
        for comp, sol in zip(completions, solution):
            content = comp[0]['content'] if isinstance(comp, list) else comp
            
            try:
                gt_data = json.loads(sol)
                gt_type = gt_data.get("异常类型", "")
                gt_risk = gt_data.get("风险等级", "")
            except:
                gt_type, gt_risk = "", ""

            pred_type, pred_risk = "", ""
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content.strip()
            
            try:
                data = json.loads(json_str)
                assessment = data["event_analyses"][0]["comprehensive_assessment"]
                pred_type = assessment.get("异常类型", "")
                pred_risk = assessment.get("风险等级", "")
            except:
                pass

            score = 0.0
            if pred_type == gt_type and gt_type != "":
                score += 1.0
            if pred_risk == gt_risk and gt_risk != "":
                score += 1.0
                
            if "detailed_cot" in content and len(content) > 150:
                score += 0.1 

            rewards.append(score)
        return rewards

# 将写好的类注册到系统的 orms 字典中（注意：右边传类名，不要带括号）
orms['my_format_reward'] = FormatRewardFunction
orms['my_accuracy_reward'] = AccuracyRewardFunction