import os
from .clip_encoder import CLIPVisionTower
from .geo_encoder import PositionalGeoEmbedding, GraphGeoModule#, SimpleGeoModule
from .entity_encoder import EntityEncoder
from .mm_retriever import MMRetriever

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_geo_tower(geo_tower_cfg, coordinates, **kwargs):
    geo_tower = getattr(geo_tower_cfg, 'geo_tower')
    if geo_tower == 'positional':
        geo_tower = PositionalGeoEmbedding(coordinates)
    elif geo_tower == 'graph':
        geo_tower = GraphGeoModule(coordinates)
    #elif geo_tower == 'simple':
    #    geo_tower = SimpleGeoModule(coordinates)
    return geo_tower
        
def build_entity_tower():
    return EntityEncoder()

def build_mm_retriever(vision_tower):
    return MMRetriever(vision_tower)