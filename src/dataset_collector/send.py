import requests

class Send:

    def __init__(self):
        self.number_plate='FFF-999'

    def send_post(self,number,ocr_score,plate_score,vehicle_score,dimension):

        url="https://servidorcloud/api/plates" #Endpoint
        print(f"Numero de placa antes: {self.number_plate} ahora: {number.upper()}")
        if(self.number_plate!=number.upper() and number!="FFF-999" and len(number.upper())==7):            
            self.number_plate=number.upper()
            # Data to be sent
            data = {
                "number": number.upper(),
                "ocrscore": ocr_score,
                "platescore": plate_score,
                "vehiclescore": vehicle_score,
                "dimension": dimension
            }
            try:
                response = requests.post(url, json=data, timeout=3)
                print(response.json())
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
            
        #else:
        #    print("no se envio")
        #    pass
