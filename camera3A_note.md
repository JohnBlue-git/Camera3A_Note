# Color coordinate
* **RGB (Red, Green, Blue)**
    * This is the most common color space used in digital images and displays. It uses 3 coordinates (one for each red, green, and blue channel) to represent a color. The value of each coordinate typically ranges from 0 (no intensity) to 255 (maximum intensity).
* **HSV (Hue, Saturation, Value)**
    * This color space represents color based on its hue (think of the color itself like red, green, blue, etc.), saturation (how intense or colorful it is), and value (brightness). It also uses 3 coordinates, but the interpretation is different from RGB.
* **LAB (Lightness, A, B)**
    * This color space separates lightness information from color information. It uses 3 coordinates: L for lightness (0 for black, 100 for white), ***A*** for ***green-red***, and ***B*** for ***blue-yellow***.

# Common camera terminology
* **Exposure**
    * This define the amount of light entering the camera. ***Exposure value*** indicates the brightness of the image, zero means normal exposure.
* **White balance**
    * White balance refers to the process of adjusting the color appearance in an image to ensure that white objects appear truly white (under human perspective) under the specific lighting conditions the image was captured in. \
* **Color temperature**
    * Color temperature is a way to measure the perceived "warmth" or "coolness" of a light source. It's typically measured in Kelvin (K). It is a color theory related to human experience about color hue. \
* **Contrast**
    * Contrast refers to the difference in brightness or color between objects in an image. High contrast images have well-defined details and separation between light and dark areas. Low contrast images appear flat and lack definition.
* **Sharpness**
    * Sharpness refers to the perceived clarity and crispness of details in an image. Sharp images have well-defined edges and visible details, while blurry images lack definition and appear hazy.

![Color Temperature](/ColorTemperature.jpg)

# Common key factors in camera control
* **Aperture**
    * The opening of the lens through which light enters the camera. It's often represented as a fraction (f-number) like f/1.8, f/5.6, etc. A lower f-number indicates a wider aperture opening.
* **Shutter speed (1 / Exposure Time)**
    * The duration for which the camera's shutter remains open, allowing light to reach the sensor. It's typically measured in fractions of a second (e.g., 1/125s, 1/2s) or many seconds.
* **Camera ISO sensitivity (sensitivity of negative film / sensor)**
    * The sensor's sensitivity to light. A higher ISO setting makes the sensor more sensitive to light, allowing you to capture images in low-light conditions without needing a slower shutter speed or wider aperture.

# Camera (3A) Tuning
* **Camera tuning**
    * It is a process to adjusting the parameters of Camera to optimize Camera’s 3A abilities, or to pretend the user preferences.

# Camera 3A:
* **Auto Exposure (AE)**
    * Auto control the amount of light captured by the sensor by controlling shutter speed, aperture, and gain (ISO). Usually, the guideline relation is similar to: \
    ```
    Exposure Value = log2(Aperture^2 * Exposure Time)
    Exposure Value = mean(image gray value)
    ```
    * But in real situation, AE algorithm will also consider about Metering Mode (ex: center-weighted, spot metering), ISO Sensitivity, and other special situation (ex: high contrast environment).

* **Auto Focus (AF)**
    * When the camera focus is adjustable, we have to apply AF to ensures that the subject is sharply focused. The AF algorithm will involves distance measurement, lens movement control, and focus confirmation. Here are mainly two approaches.
    * **Contrast-based AF**
    Finds areas of high contrast to determine focus. The camera will need to keep capturing image and examining the contrast value (ex: sharpness, edge strength) until reach the highest value.
    ```python
    import cv2

    def measure_sharpness(image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Laplacian Variance Method
        # Compute the Laplacian of the image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Calculate the variance of the Laplacian
        variance = laplacian.var()
        return variance
        
        或是
        
        # Gradient-based Metrics
        # Calculate gradient in x and y directions using Sobel operators
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        # Calculate magnitude of gradients
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        # Calculate sharpness as the mean of gradient magnitudes
        sharpness = np.mean(gradient_magnitude)
        return sharpness

    # Load an image (replace 'image.jpg' with your image file)
    image = cv2.imread('image.jpg')

    # Measure sharpness
    sharpness = measure_sharpness(image)

    print(f"Sharpness measure (Laplacian variance): {sharpness}")
    ```
    * ***Phase-detect AF:*** Measures phase shift of light rays from the subject to achieve precise focus. A flash will be emitted to target and reflects from it, and the AF sensor will sensing the difference among the reflections on multi-sensors and trigger the movements of lens.

    |Feature|Contrast-based AF|Phase-detection AF|
    | :---: | :---: | :---: |
    |Speed|Slower|Faster|
    |Accuracy|High for still subjects|Better overall|
    |Subject tracking|Struggles with moving subjects|Better at tracking moving subjects|
    |Complexity|Simple|Complex|
    |Cost|Inexpensive|Expensive|

    ![AF](/AF.jpeg)

    reference: \
    https://steemit.com/photography/@apteacher/digital-photography-focus-settings

* **Auto White Balance (AWB)**
    * Adjusts color temperature to create a natural-looking image. AWB algorithms analyze the color cast in a scene and compensate accordingly. There are some classification about (auto) white balance:
    * **Gray World Assumption**
        * It is an assumption that Ideal neutral colors would have balanced amount of R, G, and B values. However, human have also researched that the color of light would influence the description in human eyes. Therefore, a revised color temperature based AWB wouuld be more practical.
    * **White Patch Tracking**
        * Locates a white patch (or near white patch) in the image and adjusts color balance.
    * **Artistic Adjustments**
        * the normal AWS might not be desirable for artistic expression, the algorithm can be modified according to the preference of the user.
    * **Other Algorithms**
        * techniques that consider scene analysis, object recognition, and machine learning to improve accuracy beyond basic color temperature estimation.

Here is a simple color temperature based **WB code**. It is suitable to used if the image color temperature should be normal, not warm or cool. The concept is to correct the image back to normal balance between **blue/yellow** and **green/red**. (**Note:** There have some **limitations** about this white balance method):
* **Artificial lights**
    * have a non-continuous spectrum. Kelvin, based on a theoretical blackbody radiator, might not accurately represent the actual color cast of these lights.
* For artist, **white balance are often subjective**. For the sake of user preference and adaptability, modern cameras will tend to offer white balance presets and AWB algorithms instead of directly measuring Kelvin.

```python
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow

def dynamic_white_balance(image, gray_reference=(128, 128, 128)):
  """
  Attempts to perform dynamic white balance on a color image using a gray reference.

  Args:
      image (ndarray): The color image in BGR format.
      gray_reference (tuple, optional): The ideal average RGB values for neutral gray (default: (128, 128, 128)).

  Returns:
      ndarray: The white-balanced image in BGR format.
  """

  # Convert image to LAB color space
  lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  # Extract L, A, and B channels
  L, A, B = cv2.split(lab_image)

  # Calculate average values of A and B channels
  avg_A = np.mean(A)
  avg_B = np.mean(B)

  # Calculate adjustments based on gray reference
  adjustment_A = gray_reference[1] - avg_A
  adjustment_B = gray_reference[2] - avg_B

  # Adjust A and B channels (consider clamping for valid LAB values)
  A = np.clip(A + adjustment_A, 0, 255).astype(np.uint8)
  B = np.clip(B + adjustment_B, 0, 255).astype(np.uint8)

  # Merge adjusted channels back into LAB space
  lab_adjusted = cv2.merge((L, A, B))

  # Convert back to BGR color space
  return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

# Example usage (replace with your image path and adjust gray reference if needed)
# Fetch the image from the URL using requests
url = 'https://grist.org/wp-content/uploads/2021/08/twilightfilm.jpeg'
url = 'https://www.apertureacademy.com/img/how-to/3500-bridge.jpg'
response = requests.get(url)
image = np.asarray(bytearray(response.content), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
gray_reference = (127, 127, 127)  # Adjust if you have a specific reference

# white balance
white_balanced_image = dynamic_white_balance(image, gray_reference)

# Display original and white-balanced images
combine = cv2.hconcat([image, white_balanced_image])
cv2_imshow(combine)
```
![Blue](/Blue.png)
![Bridge](/Bridge.png)

\
\
\
Other urls: \
url = 'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8YmlyZHxlbnwwfHwwfHx8Mg%3D%3D'
\
url = 'https://images.unsplash.com/photo-1494500764479-0c8f2919a3d8?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
\
url = 'https://pics.craiyon.com/2023-05-27/bb48b76c0d294f3886a6134631c683d4.webp'
\
url = 'https://www.blockbluelight.com.au/cdn/shop/products/blockbluelight-blue-light-free-lighting-sweet-dreams-sleep-lights-screw-bayonet-22864746086574_1000x.jpg?v=1648873629'
\
url = 'https://grist.org/wp-content/uploads/2021/08/twilightfilm.jpeg'
\
url = 'https://images.unsplash.com/photo-1532274402911-5a369e4c4bb5?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
\
url = 'https://images.unsplash.com/photo-1523712999610-f77fbcfc3843?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
\
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRJNfSKEHCuBC5a5MFJlx4ERlnQ5mxz8a3elg&s'
\
url = 'https://photographylife.com/wp-content/uploads/2016/01/White-Balance-Correct-vs-Incorrect.jpg'
\
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScoU5q4aoYaspGnJylAdJTCFtFPMNHTiG4OQ&s'
\
url = 'https://www.apertureacademy.com/img/how-to/3500-bridge.jpg'
