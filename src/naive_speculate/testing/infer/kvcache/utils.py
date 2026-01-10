def get_num_tokens_crop(num_tokens: int, crop_ratio: float) -> int:
    return 1 if crop_ratio < 0 else int(num_tokens * crop_ratio)
