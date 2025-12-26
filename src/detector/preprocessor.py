import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Enhanced Preprocessor matching high-accuracy architecture.
    Pipeline:
    1. Drop irrelevant columns
    2. Impute missing values (Median)
    3. Scale features (StandardScaler)
    4. Feature Selection (VarianceThreshold + SelectKBest)
    """
    
    # Features to drop based on user's script
    DROP_FEATURES = [
        'Flow ID', 'Source IP', 'Destination IP', 'Timestamp',
        'Source Port', 'Destination Port', 'Flow Duration',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Rnd',
        'Fwd Header Length.1', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Flow IAT Min',
        'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Type' # Keeping original drops too just in case
    ]
    
    def __init__(self, use_multi_label=True):
        self.use_multi_label = use_multi_label
        
        # Pipeline components
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.variance_selector = VarianceThreshold(threshold=0.01)
        self.k_best_selector = SelectKBest(score_func=f_classif, k=50) # Approx top 50
        self.label_encoder = LabelEncoder()
        
        self.feature_names = None
        self.selected_features = None
        
    def load_data(self, filepath):
        """Load data from CSV or Excel"""
        logger.info(f"Loading data from {filepath}...")
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            logger.info(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def prepare_features_and_labels(self, df, training=False):
        """
        Prepare X and y.
        If training=True, fits the pipeline components.
        If training=False, transforms using existing components.
        """
        
        # 1. Clean Column Names (Remove Unnamed)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # 2. Drop irrelevant columns
        # Flexible drop: drop whatever exists from the list
        cols_to_drop = [c for c in self.DROP_FEATURES if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # 3. Handle Labels (Target)
        y = None
        if 'Label' in df.columns:
            # User specific encoding: PortScan=2, BENIGN=0, DDoS=1
            # We map this to a standardized 'Encryption' field logic or just reuse Label
            # The original code filtered 'Label' against ATTACK_CLASSES. 
            # The USER SCRIPT uses 'Label' directly for classification.
            
            # Let's align with User Script Logic
            mask = df['Label'].isin(['PortScan', 'BENIGN', 'DDoS'])
            
            if mask.any():
                df = df[mask].copy() # Filter to only these 3
                # Create numeric target
                # Map: BENIGN->0, DDoS->1, PortScan->2
                label_map = {'BENIGN': 0, 'DDoS': 1, 'PortScan': 2}
                y = df['Label'].map(label_map).fillna(-1).values
            else:
                logger.warning("Standard target labels (PortScan, BENIGN, DDoS) not found. Falling back to generic LabelEncoder.")
                # Fallback: Encode whatever labels are present
                self.label_encoder.fit(df['Label'])
                y = self.label_encoder.transform(df['Label'])
            
            # For compatibility with legacy "y_multi", we can return this as y
            # We skip specific string-based label encoder if we use this mapping
        
        # 4. Numeric Conversion & Cleaning
        X = df.select_dtypes(include=[np.number])
        # Force numeric conversion just in case
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if training:
            # === FIT PIPELINE ===
            
            # 5. Impute
            X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            
            # 6. Scale
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X_imputed), columns=X.columns)
            
            # 7. Variance Threshold
            X_reduced = self.variance_selector.fit_transform(X_scaled)
            kept_cols_var = X_scaled.columns[self.variance_selector.get_support()]
            X_reduced_df = pd.DataFrame(X_reduced, columns=kept_cols_var)
            
            # 8. Select K Best
            # We need y for this. If y is None (unlabeled data), we skip or fail.
            if y is not None:
                # Limit k to actual number of features if less than 50
                k = min(50, len(kept_cols_var))
                self.k_best_selector.k = k
                
                X_final = self.k_best_selector.fit_transform(X_reduced_df, y)
                final_cols = X_reduced_df.columns[self.k_best_selector.get_support()]
                
                self.feature_names = final_cols.tolist()
                logger.info(f"✓ Feature selection: {X.shape[1]} -> {X_final.shape[1]} features")
                
                return X_final, y, y # Returning y twice to match (X, y_binary, y_multi) signature if needed
            else:
                # If training but no y, we cannot select features based on classif.
                # Fallback: just keep variance threshold features
                self.feature_names = kept_cols_var.tolist()
                logger.warning("Training without labels! Skipping SelectKBest.")
                return X_reduced, None, None
            
        else:
            # === TRANSFORM ONLY ===
            if self.feature_names is None:
                logger.error("Preprocessor not fitted! Call with training=True first.")
                return None, None, None
                
            try:
                # Ensure we have the same columns as fit time for imputer
                # This is tricky if input df has different columns.
                # We expect roughly same structure.
                
                # 5. Impute
                # We need to ensure X has all columns expected by imputer
                # For now assume input matches training schema
                X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
                
                # 6. Scale
                X_scaled = pd.DataFrame(self.scaler.transform(X_imputed), columns=X.columns)
                
                # 7. Variance Threshold
                X_reduced = self.variance_selector.transform(X_scaled)
                kept_cols_var = X_scaled.columns[self.variance_selector.get_support()]
                X_reduced_df = pd.DataFrame(X_reduced, columns=kept_cols_var)
                
                # 8. Select K Best
                X_final = self.k_best_selector.transform(X_reduced_df)
                
                return X_final, y, y
                
            except Exception as e:
                logger.error(f"Transform error: {e}")
                # Fallback: try to return aligned zero-padded if shape mismatch?
                # For now raise
                raise e

    # Helper for compatibility
    def fit_scaler(self, X):
        pass # Now handled in prepare_features_and_labels(training=True)

    def transform(self, X):
        return X # Already transformed in prepare

