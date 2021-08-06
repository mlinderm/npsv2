def as_scalar(item_or_sequence):
    if isinstance(item_or_sequence, (list, tuple)):
        if len(item_or_sequence) != 1:
            raise ValueError("Can't convert multi-element sequence to scalar")
        return item_or_sequence[0]
    else:
        return item_or_sequence