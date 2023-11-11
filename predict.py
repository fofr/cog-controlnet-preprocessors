import torch
from typing import List
from cog import BasePredictor, Input, Path
from PIL import Image
from io import BytesIO
from controlnet_aux.processor import Processor
from controlnet_aux import (
    HEDdetector,
    MidasDetector,
    MLSDdetector,
    OpenposeDetector,
    PidiNetDetector,
    NormalBaeDetector,
    LineartDetector,
    LineartAnimeDetector,
    CannyDetector,
    ContentShuffleDetector,
    ZoeDetector,
    MediapipeFaceDetector,
    SamDetector,
    LeresDetector,
    DWposeDetector,
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.annotators = {
            "canny": CannyDetector(),
            "content": ContentShuffleDetector(),
            "face_detector": MediapipeFaceDetector(),
            "hed": self.initialize_detector(HEDdetector),
            "midas": self.initialize_detector(MidasDetector),
            "mlsd": self.initialize_detector(MLSDdetector),
            "open_pose": self.initialize_detector(OpenposeDetector),
            "pidi": self.initialize_detector(PidiNetDetector),
            "normal_bae": self.initialize_detector(NormalBaeDetector),
            "lineart": self.initialize_detector(LineartDetector),
            "lineart_anime": self.initialize_detector(LineartAnimeDetector),
            # "zoe": self.initialize_detector(ZoeDetector),
            "sam": self.initialize_detector(
                SamDetector,
                model_name="ybelkada/segment-anything",
                subfolder="checkpoints",
            ),
            # "mobile_sam": self.initialize_detector(
            #     SamDetector,
            #     model_name="dhkim2810/MobileSAM",
            #     model_type="vit_t",
            #     filename="mobile_sam.pt",
            # ),
            "leres": self.initialize_detector(LeresDetector),
        }

        torch.device("cuda")

    def initialize_detector(
        self, detector_class, model_name="lllyasviel/Annotators", **kwargs
    ):
        return detector_class.from_pretrained(
            model_name,
            cache_dir="model_cache",
            **kwargs,
        )

    def process_image(self, image, annotator):
        print(f"Processing image with {annotator}")
        return self.annotators[annotator](image)

    def predict(
        self,
        image: Path = Input(
            description="Image to preprocess",
        ),
        canny: bool = Input(
            default=True,
            description="Run canny edge detection",
        ),
        content: bool = Input(
            default=True,
            description="Run content shuffle detection",
        ),
        face_detector: bool = Input(
            default=True,
            description="Run face detection",
        ),
        hed: bool = Input(
            default=True,
            description="Run HED detection",
        ),
        midas: bool = Input(
            default=True,
            description="Run Midas detection",
        ),
        mlsd: bool = Input(
            default=True,
            description="Run MLSD detection",
        ),
        open_pose: bool = Input(
            default=True,
            description="Run Openpose detection",
        ),
        pidi: bool = Input(
            default=True,
            description="Run PidiNet detection",
        ),
        normal_bae: bool = Input(
            default=True,
            description="Run NormalBae detection",
        ),
        lineart: bool = Input(
            default=True,
            description="Run Lineart detection",
        ),
        lineart_anime: bool = Input(
            default=True,
            description="Run LineartAnime detection",
        ),
        sam: bool = Input(
            default=True,
            description="Run Sam detection",
        ),
        leres: bool = Input(
            default=True,
            description="Run Leres detection",
        ),
    ) -> List[Path]:
        # Load image
        image = Image.open(image)

        paths = []
        annotator_inputs = {
            "canny": canny,
            "content": content,
            "face_detector": face_detector,
            "hed": hed,
            "midas": midas,
            "mlsd": mlsd,
            "open_pose": open_pose,
            "pidi": pidi,
            "normal_bae": normal_bae,
            "lineart": lineart,
            "lineart_anime": lineart_anime,
            "sam": sam,
            "leres": leres,
        }
        for annotator, run_annotator in annotator_inputs.items():
            if run_annotator:
                processed_image = self.process_image(image, annotator)
                processed_image.save(f"/tmp/{annotator}.png")
                paths.append(Path(f"/tmp/{annotator}.png"))

        return paths
