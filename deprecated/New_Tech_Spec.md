# New Tech Spec: Adaptive Channel Prediction System

## Executive Summary

Our trading system has evolved into a sophisticated **Adaptive Channel Prediction System** that dynamically analyzes market structure across multiple timeframes to predict optimal entry/exit points with variable time horizons. The system uses a hierarchical neural network that "reads the ocean layers" of market data - from fast intraday ripples to slow macro tides - to make intelligent predictions.

### Core Analogy: Reading the Ocean Layers
Just as an oceanographer studies different water layers to understand currents, our system analyzes market timeframes as "layers":
- **Fast Layer**: Intraday ripples (hours) - RSI warnings and short-term volatility
- **Medium Layer**: Swings (days) - Channel alignment and trend confirmation  
- **Slow Layer**: Macro tides (weeks+) - Long-term support/resistance and fundamental drivers

The model dynamically selects the most confident layer and projects forward accordingly, using higher layers as confirmation for longer holds.

## System Architecture

### Core Components

#### 1. Training Engine
- **File**: `train_hierarchical.py`
- **Purpose**: Trains the hierarchical LNN with adaptive projection
- **Key Features**:
  - Multi-task learning (price, bands, continuation, adaptive horizons)
  - Dynamic window sizing for channel analysis
  - Continuation label generation with multi-timeframe analysis
  - Progress tracking with clean terminal output

#### 2. Dashboard Interface
- **File**: `hierarchical_dashboard.py`
- **Purpose**: Real-time prediction display with layer interplay
- **Key Features**:
  - Adaptive projection visualization
  - Layer confidence debate display
  - Dynamic time horizon predictions
  - Interactive Streamlit interface

#### 3. Feature Extraction Engine
- **File**: `src/ml/features.py`
- **Purpose**: Extracts 495+ features including channel analysis
- **Key Features**:
  - Rolling channel calculations across 11 timeframes
  - Dynamic window sizing based on volatility
  - Continuation label generation
  - GPU acceleration support

#### 4. Model Architecture
- **File**: `src/ml/hierarchical_model.py`
- **Purpose**: Hierarchical LNN with adaptive projection head
- **Key Features**:
  - 3-layer architecture (fast/medium/slow timeframes)
  - Adaptive projection for dynamic horizons
  - Multi-task outputs (price, confidence, layer selection)
  - Liquid neural network implementation

#### 5. Dataset Management
- **File**: `src/ml/hierarchical_dataset.py`
- **Purpose**: Lazy/preloaded dataset with adaptive targets
- **Key Features**:
  - Continuation label integration
  - Adaptive projection targets
  - Memory-efficient data loading

#### 6. Data Feed System
- **File**: `src/ml/data_feed.py`
- **Purpose**: CSV data loading with validation
- **Key Features**:
  - SPY/TSLA data alignment
  - Comprehensive data validation
  - Historical buffer support

#### 7. Feature Extraction (Lazy)
- **File**: `src/ml/features_lazy.py`
- **Purpose**: Progress-tracked feature extraction
- **Key Features**:
  - tqdm progress bars
  - Memory-efficient processing
  - Continuation label integration

#### 8. Channel Calculation
- **File**: `src/linear_regression.py`
- **Purpose**: Channel detection and analysis
- **Key Features**:
  - Optimal window selection
  - Ping-pong bounce detection
  - R-squared channel quality metrics

#### 9. RSI Calculator
- **File**: `src/rsi_calculator.py`
- **Purpose**: RSI-based signal generation
- **Key Features**:
  - Oversold/overbought detection
  - Signal strength calculation

#### 10. Data Feed Base
- **File**: `src/ml/base.py`
- **Purpose**: Abstract data feed interface

## Key Innovations

### 1. Adaptive Projection
- Dynamic horizon prediction (24 bars to 2,016 bars)
- Layer confidence-based timescale selection
- Multi-task learning with variable targets

### 2. Dynamic Window Sizing
- Volatility-based lookback adjustment
- Optimal channel detection across conditions
- Memory-efficient processing

### 3. Multi-Timeframe Continuation Analysis
- 1h and 4h OHLC chunk analysis
- RSI and slope alignment checks
- Look-ahead simulation for duration/gain prediction

### 4. Layer Interplay Visualization
- Real-time confidence debate display
- Dominant layer identification
- Adaptive time horizon explanations

## Data Flow

1. **Data Loading**: CSV feeds → validation → alignment
2. **Feature Extraction**: Multi-timeframe channels + continuation labels
3. **Model Training**: Hierarchical LNN with adaptive projection
4. **Prediction**: Dynamic horizon + layer confidence outputs
5. **Visualization**: Dashboard with layer interplay display

## File Dependencies

### Core System Files:
- `train_hierarchical.py` - Main training script
- `hierarchical_dashboard.py` - Prediction dashboard
- `src/ml/hierarchical_model.py` - Model architecture
- `src/ml/features.py` - Feature extraction
- `src/ml/features_lazy.py` - Lazy feature extraction
- `src/ml/hierarchical_dataset.py` - Dataset management
- `src/ml/data_feed.py` - Data loading
- `src/linear_regression.py` - Channel calculations
- `src/rsi_calculator.py` - RSI analysis
- `src/ml/base.py` - Base interfaces

### Configuration Files:
- `config/__init__.py` - Configuration management

### Legacy Files (Not Used):
- `main.py` - Old linear regression system
- `src/data_handler.py` - Old data handling
- `src/ml/model.py` - Old single-model approach
- Various `test_*.py` files for old functionality

## Performance Characteristics

- **Training Time**: ~45-60 minutes for full channel extraction
- **Model Size**: ~10% increase with adaptive projection
- **Prediction Speed**: Real-time with layer analysis
- **Memory Usage**: Efficient with lazy loading options
- **Accuracy**: Improved with multi-timeframe confirmation

## Future Enhancements

- News sentiment integration
- Additional layer types (ultra-fast, macro)
- Ensemble methods for layer voting
- Real-time adaptation during trading</content>
<parameter name="filePath">New_Tech_Spec.md