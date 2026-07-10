"""Case selection and MAT image auto-detection for FACE4."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


FACES = ("face_002", "face_005")
PROMPTS = ("add black sunglasses", "add headphones")
MODEL_NAME = "instructpix2pix_arcface_iresnet100"


def slugify(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "_", "-", "/", "."}:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


@dataclass(frozen=True)
class Case:
    face_id: str
    prompt: str

    @property
    def slug(self) -> str:
        return f"{self.face_id}__{slugify(self.prompt)}"

    @property
    def seed(self) -> int:
        digest = hashlib.sha256(f"{self.face_id}|{self.prompt}".encode("utf-8")).hexdigest()
        return 1234 + int(digest[:8], 16) % 1_000_000


@dataclass(frozen=True)
class RunSpec:
    case: Case
    seed: int | None = None

    @property
    def model(self) -> str:
        return MODEL_NAME

    @property
    def prompt(self) -> str:
        return self.case.prompt

    @property
    def face_id(self) -> str:
        return self.case.face_id

    @property
    def run_seed(self) -> int:
        return self.case.seed if self.seed is None else int(self.seed)

    @property
    def slug(self) -> str:
        return f"edited_output_identity__{self.case.slug}"


def all_cases() -> list[Case]:
    return [Case(face_id, prompt) for face_id in FACES for prompt in PROMPTS]


def build_matrix(quick: bool = False) -> list[RunSpec]:
    cases = all_cases()
    if quick:
        return [RunSpec(cases[0])]
    return [RunSpec(case) for case in cases]


def resolve_image_path(mat_root: Path, face_id: str) -> Path:
    folder = mat_root / "data" / face_id
    if not folder.exists():
        raise FileNotFoundError(f"Missing MAT face folder: {folder}")
    for name in ("instruct_512.png", "master_1024.png", "flux_768.png"):
        path = folder / name
        if path.exists():
            return path
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if images:
        return images[0]
    raise FileNotFoundError(f"No usable image found for {face_id}. Expected instruct_512.png or another image.")


def print_resolved_cases(mat_root: Path) -> None:
    for face_id in FACES:
        path = resolve_image_path(mat_root, face_id)
        print(f"[face4] input {face_id}: {path}")
