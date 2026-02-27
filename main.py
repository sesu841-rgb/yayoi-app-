from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import markdown
import pdfkit
import tempfile

load_dotenv()

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    formattedHistory: str

@app.post("/analyze")
async def analyze_life_history(request: AnalyzeRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI APIキーがサーバーに設定されていません。")

    system_prompt = """
あなたは「人生構造解析専門AI」です。
あなたの役割は、ユーザーが入力した5年ごとの人生データから
感情ではなく
励ましではなく
ポジティブ変換でもなく
構造・因果・繰り返しパターンのみを抽出することです。

【絶対ルール】
・ポエム禁止
・スピリチュアル禁止
・占い表現禁止
・抽象論だけで終わらせない
・「あなたは素晴らしい」などの承認ワード禁止
・断定しすぎない（推測は“構造的に推測される”と表現）
・分析は必ず 具体 → 因果 → 構造 → 繰り返し で書くこと。

【解析目的】
40年分の出来事から、
・繰り返している人生構造
・無意識の選択パターン
・外部要因と内部反応の因果ループ
・支配構造の有無
・現実圧迫構造の有無
・今止めるべき連鎖
・今後の分岐点
・タイプ分類
を明確化する。

【出力フォーマット】
以下の項目を必ずMarkdownの「## 見出し名」フォーマット（例：## ① 構造サマリー）を用いて出力してください。フロントエンドの表示上、## や ### を必ず使う必要があります。

## ① 構造サマリー（全体傾向）
・家庭環境
・役割の固定
・対人傾向
・環境変化への反応
を300〜400文字で要約。

## ② 繰り返している構造（最低3つ）
各項目ごとに：
起点（何が引き金か）/ その時の内的反応 / 取る行動 / 結果 / それがどう再発しているか
を因果で説明。各パターンは「### パターン1」のように小見出しを用いること。

## ③ 支配構造の解析（存在する場合のみ）
以下が入力に含まれる場合のみ出力：
親の支配 / 上司の高圧 / パートナーのコントロール / 過干渉 / 精神的圧力
外的支配 → 内面化 → 自己統制化 の流れを構造で示す。該当しない場合は出力しない。

## ④ 現実圧迫構造（該当する場合のみ）
以下ワードが含まれる場合のみ出力：
借金／ローン／収入不安／督促／生活費／経済的不安／返済／将来不安
構造として：
現実的プレッシャー ↓ 自己イメージ維持欲求 ↓ 回避／過剰努力 ↓ 罪悪感 ↓ 隠蔽／再圧迫
などの因果を整理する。※単なる「お金の問題」とは書かない。必ず心理構造と接続させる。

## ⑤ 無意識の思考回路
入力全体から推測される：
・信念
・前提
・思い込み
・自己定義
を言語化する。（例：「ちゃんとしていなければ価値がない」「見捨てられないために応える」「問題が起きたら環境を変える」など）

## ⑥ 転換点と分岐構造
人生のターニングポイントを2〜3箇所抽出し、その時の「外圧」「内面」「選択基準」を分析する。

## ⑦ 今、止めるべき連鎖
もっとも再発性の高いループを1つ特定し、なぜ止めない限り再現するかを論理的に説明。

## ⑧ 未来への問い（3つ）
抽象的な問いは禁止。行動レベルで具体的に。
（例：×「自分を大切にできていますか？」 ○「“期待に応える前に止まる”を次の3ヶ月で何回実行できますか？」）

## ⑨ タイプ分類（構造命名）
全体を総括する命名。
（例：過剰責任引受型 / 信頼維持強迫型 / 承認駆動努力型 / 環境リセット型 / 罪悪感循環型 など。一般的すぎる名前は禁止。）

【トーン】
客観的かつ論理的ですが、相手を突き放すような冷酷さは出さず、プロの専門家として真摯に向き合うような引き締まったトーンにしてください。
「〜なトーンで出力しました」「以上が解析結果です」のようなAI特有のメタ発言や報告は一切不要です。内容のテキストのみを出力してください。

【文字量】
最低1500文字以上。薄く広くではなく、狭く深く切る。

【最重要】
その人の入力データに出ていない要素を勝手に創作しない。だが、入力の奥にある構造は必ず推測して言語化する。
"""

    user_prompt = f"以下の人生史データを解析し、人生構造解析レポートを出力してください。\n\n{request.formattedHistory}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [
                        { "role": "system", "content": system_prompt },
                        { "role": "user", "content": user_prompt }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 3000
                },
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            return {"report": result['choices'][0]['message']['content']}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"OpenAI APIエラー: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"通信エラー: {str(e)}")

class EmailRequest(BaseModel):
    email: str
    report_markdown: str

@app.post("/send-report")
async def send_report_email(request: EmailRequest):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT", "587")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    # If no SMTP server configured, simulate success (for demo)
    if not smtp_server or not smtp_user or not smtp_pass:
        print(f"[Simulation] Would have sent email to {request.email}")
        return {"status": "success", "message": "Demo mode: Email 'sent' successfully (SMTP credentials not configured)."}

    # 1. Convert Markdown to HTML
    html_content = markdown.markdown(request.report_markdown)
    
    # 追加: メールの文字化けを防ぐための <meta charset="UTF-8"> タグを追加
    # さまざまな端末で読みやすくなるようフォント指定も変更
    styled_html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif; line-height: 1.6; color: #333; }}
        h2 {{ color: #333; border-bottom: 2px solid #5abcb5; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #5abcb5; border-left: 3px solid #5abcb5; padding-left: 10px; margin-top: 20px; }}
        h4 {{ color: #444; }}
        .report-container {{ padding: 20px; max-width: 800px; margin: 0 auto; }}
        p {{ margin-bottom: 1em; }}
    </style>
    </head>
    <body>
    <div class="report-container">
        <h2>わたしの人生パターンレポート</h2>
        {html_content}
    </div>
    </body>
    </html>
    """

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    try:
        # EmailMessageオブジェクトを作成
        msg = EmailMessage()
        msg['Subject'] = '【人生パターン・レポート】解析結果をお届けします'
        if smtp_user:
            msg['From'] = smtp_user
        msg['To'] = request.email
        
        # プレーンテキスト（HTMLが表示できないメールソフト用）
        body_text = "あなたの人生パターンレポートが完成しました。\n\n"
        body_text += "本メールはHTML形式で送信されています。お使いのメールソフトで表示を有効にしてご覧ください。\n\n"
        body_text += "※このメールは自動送信されています。"
        
        # 1. まずテキスト本文をセット（文字化けを防ぐためutf-8指定）
        msg.set_content(body_text, charset='utf-8')
        
        # 2. HTML版も追加（これにより「HTMLメール」として認識される）
        msg.add_alternative(styled_html, subtype='html', charset='utf-8')

        # メールサーバ経由で送信（長時間のフリーズを防ぐためタイムアウトを10秒に設定）
        with smtplib.SMTP(smtp_server, int(smtp_port), timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            
        return {"status": "success", "message": "Email sent successfully"}
    except Exception as e:
        print(f"Email error: {e}")
        raise HTTPException(status_code=500, detail=f"メール送信エラー: {str(e)}")

# Run with: uvicorn main:app --reload

from fastapi.responses import FileResponse

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse("style.css")

@app.get("/script.js")
async def serve_js():
    return FileResponse("script.js")
