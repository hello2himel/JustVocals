from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, FileField
from wtforms.validators import Regexp, Optional


class ProcessForm(FlaskForm):
    link = StringField('YouTube URL', validators=[
        Optional(),
        Regexp(r'https?://(www\.)?(youtube\.com|youtu\.be)/', message="Invalid YouTube URL")
    ])
    audio_file = FileField('Upload Audio File', validators=[Optional()])
    remove_silence = BooleanField('Remove Silence', default=True)
    enhance_vocals = BooleanField('Enhance Vocals', default=True)
    submit = SubmitField('Remove Instruments')
