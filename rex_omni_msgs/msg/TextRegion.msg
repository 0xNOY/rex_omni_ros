# Single recognized text region.
# bbox is filled for the WORD_BOX / TEXTLINE_BOX variants,
# polygon is filled for the POLYGON variant.
string text
BoundingBox bbox
Polygon2D polygon
float32 confidence  # pseudo-confidence from token logprobs; 0 when disabled
