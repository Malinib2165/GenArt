import requests

def test_effect(effect_name):
    url = "http://127.0.0.1:5000/apply_effect"
    files = {'file': open('test_image.png', 'rb')}
    data = {'effect': effect_name}
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        img_data = response.json().get('processed_image')
        if img_data:
            header, encoded = img_data.split(',', 1)
            with open(f'output_{effect_name}.png', 'wb') as f:
                f.write(base64.b64decode(encoded))
            print(f"{effect_name} effect test passed, output saved as output_{effect_name}.png")
        else:
            print(f"{effect_name} effect test failed: No image data in response")
    else:
        print(f"{effect_name} effect test failed: Status code {response.status_code}")

if __name__ == "__main__":
    import base64
    for effect in ['pencil_gray', 'oil_painting', 'watercolor']:
        test_effect(effect)
