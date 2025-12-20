from app.graph.state import GraphState
from app.services.llm_helper import call_text_llm

def socratic_node(state: GraphState) -> GraphState:
    if state.analysis is None or not hasattr(state.analysis, "analysis"):
        raise ValueError("state.analysis is None or missing 'analysis' attribute")
    current_analysis = state.analysis.analysis
    current_assumptions = state.analysis.assumptions
    current_uncertainties = state.analysis.uncertainties
    
    raw_text = call_text_llm(
        task_name="socratic_question",
        prompt=f"""
        TASK: Critical Thinking Assistant
        You are a Socratic questioning specialist designed to deepen analytical thinking through targeted inquiry.
        Core Responsibility:
        Generate a single, powerful Socratic question that challenges the user to examine their reasoning more rigorously.
        Input:

        {current_analysis} - The user's current thinking, argument, or analysis
        {current_assumptions} - A list of assumptions underlying the user's analysis
        {current_uncertainties} - A list of uncertainties or knowledge gaps in the user's analysis

        Question Design Principles:

        Specificity Over Generality

        Root the question directly in concrete details from the analysis
        Avoid generic prompts like "Have you considered other perspectives?"
        Reference specific claims, assumptions, or reasoning steps


        Cognitive Challenge Targets

        Assumptions: Unexamined beliefs treated as given
        Evidence quality: Strength, relevance, and sufficiency of support
        Logic gaps: Leaps in reasoning or unstated connections
        Alternative explanations: Competing hypotheses not considered
        Scope boundaries: What's been excluded or overlooked
        Implications: Logical consequences not fully explored


        Question Characteristics

        Open-ended: Cannot be answered with simple yes/no
        Provocative: Creates productive cognitive dissonance
        Clear: Immediately understandable without clarification
        Focused: Addresses one specific aspect deeply rather than many superficially
        Encourage referencing a specific assumption or uncertainty



        Output Format:
        Return only the question itself—no preamble, explanation, or meta-commentary. 
        It should be a single, concise sentence ending with a question mark.
        """)
    state.socratic_question = raw_text.strip()
    return state
    