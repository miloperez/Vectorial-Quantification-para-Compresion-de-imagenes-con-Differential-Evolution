import numpy as np
import ImageTools as IT
import CuantiVect as CV
from tqdm import tqdm


for k in tqdm(["Banca", "nye"]):
    file = k + ".jpg"

    IT.displayBWImage(IT.loadImgBW(file), "Original")
    img = IT.loadImgBW(file)
    goal = np.shape(img)

    img = np.reshape(img, (int(pow(len(img), 0.5)), int(pow(len(img), 0.5))))

    for i in tqdm([2, 4, 5, 10]):
        for j in tqdm([2, 4, 8, 16, 32, 64, 128]):
            img_1 = CV.VectorialQuantification(i, j, img)
            img_2 = img_1.getImage()
            img_2 = np.reshape(img_2, goal)

            # IT.displayBWImage(img_2, f"Bloque {i}x{i} Codewords {j}")
            IT.saveBWImage(img_2, f"{k} Bloque {i}x{i} Codewords {j}")