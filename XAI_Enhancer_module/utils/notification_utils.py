
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_email_notification(to_email: str, subject: str, body: str, 
                          sender_email: str = None, sender_password: str = None,
                          smtp_server: str = "smtp.gmail.com", smtp_port: int = 587,
                          subtype: str = "plain"):
    """
    Send an email notification.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body content
        sender_email: Sender email address (defaults to env var SENDER_EMAIL)
        sender_password: Sender password or app password (defaults to env var SENDER_PASSWORD)
        smtp_server: SMTP server address (default: smtp.gmail.com)
        smtp_port: SMTP port (default: 587 for TLS)
        subtype: MIME text subtype ('plain' or 'html')
    
    Returns:
        bool: True if sent successfully, False otherwise
    """
    if not sender_email:
        sender_email = os.environ.get("SENDER_EMAIL")
    
    if not sender_password:
        sender_password = os.environ.get("SENDER_PASSWORD")
        
    if not sender_email or not sender_password:
        print("❌ Error: Sender email and password must be provided via arguments or environment variables (SENDER_EMAIL, SENDER_PASSWORD)")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, subtype))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        
        print(f"✅ Notification email sent to {to_email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
