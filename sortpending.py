import base64
import io
import os
import shutil

from openai import OpenAI

import rawpy
from PIL import Image

client = OpenAI()

# List all files and directories in the specified path
path = 'data/RAW/SC1/BR1'
pending = os.path.join(path, '00570')

bitting = None

for file in sorted(os.listdir(pending)):
    if file.startswith('.'):
        continue

    print(file)
    filename = os.path.join(pending, file)

    raw = rawpy.imread(filename).postprocess()
    rgb = Image.fromarray(raw).convert('RGB')
    smaller = rgb.resize((512, 384))

    buf = io.BytesIO()
    smaller.save(buf, format='JPEG')
    blob_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'If this is a picture of a white card with black lettering on it, respond with just the 5 digit number written on the card. If this is a picture of a key, respond with "front" if you can see the engravings "SC1" & "USA", and "back" if you cannot.'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{blob_b64}'
                        }
                    }
                ]
            }
        ],
        model='gpt-4o',
    )

    content = chat_completion.choices[0].message.content

    print(content)

    match content:
        case 'front':
            assert bitting is not None, 'key image before card image'
            dest = os.path.join(path, bitting, 'front.orf')
            assert not os.path.exists(dest), 'key image already exists'
            shutil.move(filename, dest)

        case 'back':
            assert bitting is not None, 'key image before card image'
            dest = os.path.join(path, bitting, 'back.orf')
            assert not os.path.exists(dest), 'key image already exists'
            shutil.move(filename, dest)

        case _:
            bitting = content
            os.makedirs(os.path.join(path, bitting), exist_ok=True)
            dest = os.path.join(path, bitting, 'card.orf')
            assert not os.path.exists(dest), 'card image already exists'
            shutil.move(filename, dest)