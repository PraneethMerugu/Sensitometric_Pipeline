"""
Configuration management for the Film Emulation Pipeline.
Defines the schema for film profiles and handles loading from JSON.
Refactored to use Equinox Modules for differentiability.
"""

import json
import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import dataclasses

class OpticalConfig(eqx.Module):
    scatter_gamma: Union[float, jax.Array]
    bloom_weight: Union[float, jax.Array]
    halation_radius: Union[float, jax.Array]
    halation_sigma: Union[float, jax.Array]
    halation_gain: Union[float, jax.Array]
    yellow_filter_cutoff: Union[float, jax.Array]
    yellow_filter_slope: Union[float, jax.Array]
    halation_smoothness: Union[float, jax.Array]

    def __init__(
        self,
        scatter_gamma: float = 0.65,
        bloom_weight: float = 0.15,
        halation_radius: float = 30.0,
        halation_sigma: float = 8.0,
        halation_gain: float = 1.5,
        yellow_filter_cutoff: float = 510.0,
        yellow_filter_slope: float = 0.15,
        halation_smoothness: float = 0.2
    ):
        self.scatter_gamma = jnp.array(scatter_gamma)
        self.bloom_weight = jnp.array(bloom_weight)
        self.halation_radius = jnp.array(halation_radius)
        self.halation_sigma = jnp.array(halation_sigma)
        self.halation_gain = jnp.array(halation_gain)
        self.yellow_filter_cutoff = jnp.array(yellow_filter_cutoff)
        self.yellow_filter_slope = jnp.array(yellow_filter_slope)
        self.halation_smoothness = jnp.array(halation_smoothness)

class SensitometryConfig(eqx.Module):
    # Curve parameters can be provided directly or loaded from CSVs
    # If provided, expected shape (3, 5) flattened or list of lists
    curve_params: Optional[Union[List[List[float]], jax.Array]]
    
    # Paths (Static fields -> Not differentiable)
    # We use eqx.field(static=True) for these
    red_curve_path: Optional[str] = eqx.field(static=True, default=None)
    green_curve_path: Optional[str] = eqx.field(static=True, default=None)
    blue_curve_path: Optional[str] = eqx.field(static=True, default=None)

    def __init__(
        self,
        curve_params: Optional[List[List[float]]] = None,
        red_curve_path: Optional[str] = None,
        green_curve_path: Optional[str] = None,
        blue_curve_path: Optional[str] = None
    ):
        self.curve_params = jnp.array(curve_params) if curve_params is not None else None
        self.red_curve_path = red_curve_path
        self.green_curve_path = green_curve_path
        self.blue_curve_path = blue_curve_path

class ChemicalConfig(eqx.Module):
    sigma_soft: Union[float, jax.Array]
    sigma_hard: Union[float, jax.Array]
    gamma: Union[float, jax.Array]
    coupling_matrix: Optional[Union[List[List[float]], jax.Array]]
    d_min: Union[float, jax.Array]
    d_max: Union[float, jax.Array]
    drag_ratio: Union[float, jax.Array]
    exhaustion_alpha: Union[float, jax.Array]
    exhaustion_beta: Union[float, jax.Array]
    
    def __init__(
        self,
        sigma_soft: float = 2.0,
        sigma_hard: float = 0.5,
        gamma: float = 2.0,
        coupling_matrix: Optional[List[List[float]]] = None,
        d_min: float = 0.0,
        d_max: float = 3.0,
        drag_ratio: float = 1.0,
        exhaustion_alpha: float = 2.0,
        exhaustion_beta: float = 0.5
    ):
        self.sigma_soft = jnp.array(sigma_soft)
        self.sigma_hard = jnp.array(sigma_hard)
        self.gamma = jnp.array(gamma)
        if coupling_matrix is None:
            self.coupling_matrix = jnp.eye(3) * 0.5
        else:
            self.coupling_matrix = jnp.array(coupling_matrix)
        self.d_min = jnp.array(d_min)
        self.d_max = jnp.array(d_max)
        self.drag_ratio = jnp.array(drag_ratio)
        self.exhaustion_alpha = jnp.array(exhaustion_alpha)
        self.exhaustion_beta = jnp.array(exhaustion_beta)

class GrainConfig(eqx.Module):
    enabled: bool = eqx.field(static=True)
    model_path: str = eqx.field(static=True)
    grain_radius: float

    def __init__(
        self,
        enabled: bool = True,
        model_path: str = "data/grainnet/grainnet_flax.pkl",
        grain_radius: float = 0.5
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.grain_radius = grain_radius

class PathsConfig(eqx.Module):
    # All fields here are static since they are paths
    base_dir: str = eqx.field(static=True, default=".")
    lut_dir: str = eqx.field(static=True, default="data/luts")
    
    cyan_density: str = eqx.field(static=True, default="data/Kodak_Vision3_250d/cyan_density.csv")
    magenta_density: str = eqx.field(static=True, default="data/Kodak_Vision3_250d/magenta_density.csv")
    yellow_density: str = eqx.field(static=True, default="data/Kodak_Vision3_250d/yellow_density.csv")
    min_density: str = eqx.field(static=True, default="data/Kodak_Vision3_250d/minimum_density.csv")
    
    status_m_red: str = eqx.field(static=True, default="data/StatusM/StatusM_Red.csv")
    status_m_green: str = eqx.field(static=True, default="data/StatusM/StatusM_Green.csv")
    status_m_blue: str = eqx.field(static=True, default="data/StatusM/StatusM_Blue.csv")
    
    def __init__(self, **kwargs):
        # We need a custom __init__ to handle dict-unpacking if kwargs passed
        # Or standard init
        for k, v in kwargs.items():
            if hasattr(self, k):
                object.__setattr__(self, k, v)


class FilmConfig(eqx.Module):
    name: str = eqx.field(static=True)
    optical: OpticalConfig
    sensitometry: SensitometryConfig
    chemical: ChemicalConfig
    grain: GrainConfig
    paths: PathsConfig

    def __init__(
        self,
        name: str = "Default Film",
        optical: Optional[OpticalConfig] = None,
        sensitometry: Optional[SensitometryConfig] = None,
        chemical: Optional[ChemicalConfig] = None,
        grain: Optional[GrainConfig] = None,
        paths: Optional[PathsConfig] = None
    ):
        self.name = name
        self.optical = optical if optical is not None else OpticalConfig()
        self.sensitometry = sensitometry if sensitometry is not None else SensitometryConfig()
        self.chemical = chemical if chemical is not None else ChemicalConfig()
        self.grain = grain if grain is not None else GrainConfig()
        self.paths = paths if paths is not None else PathsConfig()

    @classmethod
    def from_json(cls, path: str) -> 'FilmConfig':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        config_dir = path.parent
        
        # Helper to unpack dict to module or default
        def make_module(klass, d):
             if d is None: return klass()
             return klass(**d)

        # 1. Parse Sub-Configs
        optical = make_module(OpticalConfig, data.get('optical'))
        sensitometry = make_module(SensitometryConfig, data.get('sensitometry'))
        chemical = make_module(ChemicalConfig, data.get('chemical'))
        grain = make_module(GrainConfig, data.get('grain'))
        
        # Paths - Need manual handling for resolution
        p_data = data.get('paths', {})
        paths = PathsConfig(**p_data)
        
        # Resolve Paths helper
        def resolve(p_str):
            if not p_str: return p_str
            p = Path(p_str)
            if not p.is_absolute():
                return str(config_dir / p)
            return str(p)
            
        # Update path fields manually since Modules are immutable-ish (but in __init__ we construct new)
        # Wait, we are constructing `paths` instance. We can't mutate it after.
        # We should resolve in the dict before passing to constructor.
        
        # Let's rebuild the dict
        p_dict = {
           'base_dir': str(config_dir),
           'lut_dir': resolve(p_data.get('lut_dir', 'data/luts')),
           'cyan_density': resolve(p_data.get('cyan_density', 'data/Kodak_Vision3_250d/cyan_density.csv')),
           'magenta_density': resolve(p_data.get('magenta_density', 'data/Kodak_Vision3_250d/magenta_density.csv')),
           'yellow_density': resolve(p_data.get('yellow_density', 'data/Kodak_Vision3_250d/yellow_density.csv')),
           'min_density': resolve(p_data.get('min_density', 'data/Kodak_Vision3_250d/minimum_density.csv')),
           'status_m_red': resolve(p_data.get('status_m_red', 'data/StatusM/StatusM_Red.csv')),
           'status_m_green': resolve(p_data.get('status_m_green', 'data/StatusM/StatusM_Green.csv')),
           'status_m_blue': resolve(p_data.get('status_m_blue', 'data/StatusM/StatusM_Blue.csv')),
        }
        paths = PathsConfig(**p_dict)
        
        # Also Resolve Sensitometry Paths
        s_red = resolve(sensitometry.red_curve_path) if sensitometry.red_curve_path else None
        s_green = resolve(sensitometry.green_curve_path) if sensitometry.green_curve_path else None
        s_blue = resolve(sensitometry.blue_curve_path) if sensitometry.blue_curve_path else None
        
        sensitometry = SensitometryConfig(
            curve_params=sensitometry.curve_params,
            red_curve_path=s_red,
            green_curve_path=s_green,
            blue_curve_path=s_blue
        )
        
        # Also Resolve Grain Path
        g_path = resolve(grain.model_path)
        grain = GrainConfig(
            enabled=grain.enabled,
            model_path=g_path,
            grain_radius=grain.grain_radius
        )

        return cls(
            name=data.get('name', "Custom Config"),
            optical=optical,
            sensitometry=sensitometry,
            chemical=chemical,
            grain=grain,
            paths=paths
        )
        
    def to_json(self, path: str = None) -> Optional[str]:
        """
        Serialize to JSON. If path is provided, write to file.
        Returns JSON string.
        """
        
        # Helper to convert Module to dict
        def to_dict(module):
            # We can use tree_flatten but that loses field names.
            # Best is to iterate vars() or use manual mapping for safety.
            # Or use dataclasses.asdict if we used dataclasses but we are using Modules.
            # eqx modules don't have .asdict()
            
            # Simple approach: traverse fields
            # Getting fields of instance
            d = {}
            # For dynamic fields
            # We can't easily introspect ALL fields (static + dynamic) uniformly in Equinox easily without some tricks
            # But we know our schema.
            if isinstance(module, OpticalConfig):
                d = {
                    'scatter_gamma': module.scatter_gamma,
                    'bloom_weight': module.bloom_weight,
                    'halation_radius': module.halation_radius,
                    'halation_sigma': module.halation_sigma,
                    'halation_gain': module.halation_gain,
                    'yellow_filter_cutoff': module.yellow_filter_cutoff,
                    'yellow_filter_slope': module.yellow_filter_slope,
                    'halation_smoothness': module.halation_smoothness,
                }
            elif isinstance(module, SensitometryConfig):
                d = {
                    'curve_params': module.curve_params, # This might be JAX array, need to convert to list
                    'red_curve_path': module.red_curve_path,
                    'green_curve_path': module.green_curve_path,
                    'blue_curve_path': module.blue_curve_path
                }
            elif isinstance(module, ChemicalConfig):
                d = {
                    'sigma_soft': module.sigma_soft,
                    'sigma_hard': module.sigma_hard,
                    'gamma': module.gamma,
                    'coupling_matrix': module.coupling_matrix, # JAX array
                    'd_min': module.d_min,
                    'd_max': module.d_max,
                    'drag_ratio': module.drag_ratio,
                    'exhaustion_alpha': module.exhaustion_alpha,
                    'exhaustion_beta': module.exhaustion_beta
                }
            elif isinstance(module, GrainConfig):
                 d = {
                    'enabled': module.enabled,
                    'model_path': module.model_path,
                    'grain_radius': module.grain_radius
                }
            elif isinstance(module, PathsConfig):
                 # We probably don't want to save absolute paths if we are sharing...
                 # But simplistic "save current state" is fine.
                 d = {
                     'base_dir': module.base_dir,
                     'lut_dir': module.lut_dir,
                     'cyan_density': module.cyan_density,
                     'magenta_density': module.magenta_density,
                     'yellow_density': module.yellow_density,
                     'min_density': module.min_density,
                     'status_m_red': module.status_m_red,
                     'status_m_green': module.status_m_green,
                     'status_m_blue': module.status_m_blue
                 }
                 
            # Recursive conversion of JAX arrays to python types (list/float)
            out = {}
            for k, v in d.items():
                if hasattr(v, 'tolist'): # JAX array
                    out[k] = v.tolist()
                elif isinstance(v, list):
                     # Check if list contains arrays
                     new_list = []
                     for item in v:
                         if hasattr(item, 'tolist'):
                             new_list.append(item.tolist())
                         else:
                             new_list.append(item)
                     out[k] = new_list
                else:
                    out[k] = v
            return out

        data = {
            'name': self.name,
            'optical': to_dict(self.optical),
            'sensitometry': to_dict(self.sensitometry),
            'chemical': to_dict(self.chemical),
            'grain': to_dict(self.grain),
            'paths': to_dict(self.paths)
        }
        
        s = json.dumps(data, indent=4)
        
        if path:
            with open(path, 'w') as f:
                f.write(s)
                
        return s

def load_config(path: str) -> FilmConfig:
    return FilmConfig.from_json(path)
