import gradio as gr
from pages.page1_representation import idiom_representation_ui
from pages.page2_mining import idiom_mining_ui
from pages.page3_abstraction import abstraction_ui
from pages.page4_judgement import judgement_ui

# Improved color scheme for clarity in papers (high contrast, less yellow, more white/blue/gray)
module_card_style = """<div style="background:#f6f8fa;border:2px solid #3b82f6;
border-radius:18px;padding:18px 20px 10px 20px;margin-bottom:16px;
box-shadow:0 2px 8px #b3c6e0;text-align:center;">{content}</div>"""

def make_tag(text):
    return f'<span style="background:#e0e7ef;color:#1e293b;border-radius:6px;padding:2px 10px;font-size:13px;border:1px solid #3b82f6;display:inline-block;text-align:center;">{text}</span>'

with gr.Blocks(title="CIMAS Main") as main_app:
    gr.Markdown("<h1 style='text-align:center;background:#e0e7ef;border-radius:12px;padding:6px 0 6px 0;margin-bottom:18px;border:2px solid #3b82f6;color:#1e293b;'>🧿 CIMAS - Code Idiom Mining, Abstraction and Synthesis</h1>")

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(module_card_style.format(content=f"""
                <div style="display:flex;align-items:center;justify-content:center;margin-bottom:8px;">
                    <div style="background:#3b82f6;color:#fff;border:2px solid #3b82f6;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:bold;margin-right:12px;">1</div>
                    <span style="font-size:22px;font-weight:bold;text-align:center;color:#1e293b;">Idiom Representation Identification</span>
                </div>
            """))
            btn1 = gr.Button("Execute Module 1", elem_id="btn1", elem_classes="center-btn")

            gr.HTML(module_card_style.format(content=f"""
                <div style="display:flex;align-items:center;justify-content:center;margin-bottom:8px;">
                    <div style="background:#3b82f6;color:#fff;border:2px solid #3b82f6;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:bold;margin-right:12px;">3</div>
                    <span style="font-size:22px;font-weight:bold;text-align:center;color:#1e293b;">Differentiating Element Abstraction</span>
                </div>
            """))
            btn3 = gr.Button("Execute Module 3", elem_id="btn3", elem_classes="center-btn")

        with gr.Column(scale=2):
            gr.HTML(module_card_style.format(content=f"""
                <div style="display:flex;align-items:center;justify-content:center;margin-bottom:8px;">
                    <div style="background:#3b82f6;color:#fff;border:2px solid #3b82f6;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:bold;margin-right:12px;">2</div>
                    <span style="font-size:22px;font-weight:bold;text-align:center;color:#1e293b;">Idiomatic Code Mining</span>
                </div>
            """))
            btn2 = gr.Button("Execute Module 2", elem_id="btn2", elem_classes="center-btn")

            gr.HTML(module_card_style.format(content=f"""
                <div style="display:flex;align-items:center;justify-content:center;margin-bottom:8px;">
                    <div style="background:#3b82f6;color:#fff;border:2px solid #3b82f6;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:bold;margin-right:12px;">4</div>
                    <span style="font-size:22px;font-weight:bold;text-align:center;color:#1e293b;">Idiom Judgment and Synthesis</span>
                </div>
            """))
            btn4 = gr.Button("Execute Module 4", elem_id="btn4", elem_classes="center-btn")


    # Hidden subpages (use update to control visibility)
    rep_page = idiom_representation_ui()
    mine_page = idiom_mining_ui()
    abs_page = abstraction_ui()
    judge_page = judgement_ui()

    btn1.click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)), None, [rep_page, mine_page, abs_page, judge_page])
    btn2.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)), None, [rep_page, mine_page, abs_page, judge_page])
    btn3.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)), None, [rep_page, mine_page, abs_page, judge_page])
    btn4.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)), None, [rep_page, mine_page, abs_page, judge_page])

main_app.launch()

# cd CIMAS
# python3 cimas_web.py