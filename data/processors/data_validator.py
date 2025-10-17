"""
數據驗證模組
負責檢查數據完整性、異常值檢測和數據質量評估
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# 設定日誌
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """數據驗證結果"""
    is_valid: bool
    total_records: int
    valid_records: int
    issues_found: Dict[str, int]
    warnings: List[str]
    errors: List[str]

@dataclass
class DataQualityMetrics:
    """數據質量指標"""
    completeness: float  # 完整性（0-1）
    accuracy: float     # 準確性（0-1）
    consistency: float  # 一致性（0-1）
    timeliness: float   # 及時性（0-1）
    overall_score: float  # 綜合得分（0-1）

class DataValidator:
    """數據驗證器"""

    def __init__(self, strict_mode: bool = False):
        """
        初始化驗證器

        Args:
            strict_mode: 嚴格模式，發現任何問題都視為無效
        """
        self.strict_mode = strict_mode

    def validate_ohlcv_data(
        self,
        df: pd.DataFrame,
        symbol: str = None
    ) -> ValidationResult:
        """
        驗證OHLCV數據

        Args:
            df: 要驗證的DataFrame
            symbol: 交易對符號（用於日誌）

        Returns:
            驗證結果
        """
        if df.empty:
            return ValidationResult(
                is_valid=False,
                total_records=0,
                valid_records=0,
                issues_found={},
                warnings=["DataFrame is empty"],
                errors=["No data to validate"]
            )

        symbol_info = f" for {symbol}" if symbol else ""
        logger.info(f"Validating OHLCV data{symbol_info}: {len(df)} records")

        issues_found = {}
        warnings = []
        errors = []

        # 1. 檢查必要欄位
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            errors.append(error_msg)
            if self.strict_mode:
                return ValidationResult(
                    is_valid=False,
                    total_records=len(df),
                    valid_records=0,
                    issues_found={'missing_columns': len(missing_columns)},
                    warnings=warnings,
                    errors=errors
                )

        # 2. 檢查數據類型
        type_issues = self._check_data_types(df, required_columns)
        if type_issues:
            issues_found.update(type_issues)
            warnings.extend([f"Type issue in {col}: {desc}" for col, desc in type_issues.items()])

        # 3. 檢查價格邏輯關係
        price_logic_issues = self._check_price_logic(df)
        if price_logic_issues:
            issues_found.update(price_logic_issues)
            for issue, count in price_logic_issues.items():
                warnings.append(f"Price logic issue '{issue}': {count} occurrences")

        # 4. 檢查負數值
        negative_issues = self._check_negative_values(df)
        if negative_issues:
            issues_found.update(negative_issues)
            for issue, count in negative_issues.items():
                warnings.append(f"Negative values '{issue}': {count} occurrences")

        # 5. 檢查極端值（異常值）
        outlier_issues = self._check_outliers(df)
        if outlier_issues:
            issues_found.update(outlier_issues)
            for issue, count in outlier_issues.items():
                warnings.append(f"Outliers '{issue}': {count} occurrences")

        # 6. 檢查時間戳問題
        timestamp_issues = self._check_timestamp_issues(df)
        if timestamp_issues:
            issues_found.update(timestamp_issues)
            for issue, count in timestamp_issues.items():
                warnings.append(f"Timestamp issue '{issue}': {count} occurrences")

        # 7. 檢查重複數據
        duplicate_issues = self._check_duplicates(df)
        if duplicate_issues:
            issues_found.update(duplicate_issues)
            for issue, count in duplicate_issues.items():
                warnings.append(f"Duplicates '{issue}': {count} occurrences")

        # 8. 檢查數據連續性
        continuity_issues = self._check_data_continuity(df)
        if continuity_issues:
            issues_found.update(continuity_issues)
            for issue, count in continuity_issues.items():
                warnings.append(f"Continuity issue '{issue}': {count} occurrences")

        # 計算有效記錄數
        valid_records = len(df)
        for issue_count in issues_found.values():
            if self.strict_mode:
                valid_records -= issue_count

        valid_records = max(0, valid_records)

        # 決定整體有效性
        is_valid = len(errors) == 0 and (len(warnings) == 0 or not self.strict_mode)

        logger.info(f"Validation complete{symbol_info}: "
                   f"Valid={is_valid}, Valid records={valid_records}/{len(df)}")

        return ValidationResult(
            is_valid=is_valid,
            total_records=len(df),
            valid_records=valid_records,
            issues_found=issues_found,
            warnings=warnings,
            errors=errors
        )

    def _check_data_types(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
        """檢查數據類型問題"""
        issues = {}

        for col in columns:
            if col not in df.columns:
                continue

            # 檢查時間戳
            if col == 'timestamp':
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    issues[f'{col}_type'] = len(df)

            # 檢查數值欄位
            elif col in ['open', 'high', 'low', 'close', 'volume']:
                non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
                if non_numeric > 0:
                    issues[f'{col}_non_numeric'] = non_numeric

        return issues

    def _check_price_logic(self, df: pd.DataFrame) -> Dict[str, int]:
        """檢查價格邏輯關係"""
        issues = {}

        if 'high' in df.columns and 'low' in df.columns:
            # 最高價應該 >= 最低價
            high_low_issue = (df['high'] < df['low']).sum()
            if high_low_issue > 0:
                issues['high_below_low'] = high_low_issue

        if 'high' in df.columns and 'close' in df.columns:
            # 最高價應該 >= 收盤價
            high_close_issue = (df['high'] < df['close']).sum()
            if high_close_issue > 0:
                issues['high_below_close'] = high_close_issue

        if 'low' in df.columns and 'close' in df.columns:
            # 最低價應該 <= 收盤價
            low_close_issue = (df['low'] > df['close']).sum()
            if low_close_issue > 0:
                issues['low_above_close'] = low_close_issue

        return issues

    def _check_negative_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """檢查負數值"""
        issues = {}

        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues[f'{col}_negative'] = negative_count

        # 交易量可以為0但不應該為負
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues['volume_negative'] = negative_volume

        return issues

    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """檢查異常值（使用IQR方法）"""
        issues = {}

        # 檢查價格異常值
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns and len(df[col].dropna()) > 10:
                outlier_count = self._detect_outliers_iqr(df[col])
                if outlier_count > 0:
                    issues[f'{col}_outliers'] = outlier_count

        # 檢查交易量異常值
        if 'volume' in df.columns and len(df['volume'].dropna()) > 10:
            outlier_count = self._detect_outliers_iqr(df['volume'])
            if outlier_count > 0:
                issues['volume_outliers'] = outlier_count

        return issues

    def _detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> int:
        """使用IQR方法檢測異常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers.sum()

    def _check_timestamp_issues(self, df: pd.DataFrame) -> Dict[str, int]:
        """檢查時間戳問題"""
        issues = {}

        if 'timestamp' not in df.columns:
            return issues

        # 檢查時間戳順序
        if len(df) > 1:
            # 檢查是否有遞減的時間戳
            decreasing_timestamps = (df['timestamp'].diff().dt.total_seconds() < 0).sum()
            if decreasing_timestamps > 0:
                issues['decreasing_timestamps'] = decreasing_timestamps

            # 檢查重複時間戳
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                issues['duplicate_timestamps'] = duplicate_timestamps

        # 檢查未來時間戳（超過當前時間）
        now = pd.Timestamp.now()
        future_timestamps = (df['timestamp'] > now).sum()
        if future_timestamps > 0:
            issues['future_timestamps'] = future_timestamps

        return issues

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, int]:
        """檢查重複數據"""
        issues = {}

        # 檢查完全重複的行
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues['duplicate_rows'] = duplicate_rows

        # 檢查重複的時間戳（對於同一交易對）
        if 'timestamp' in df.columns:
            timestamp_counts = df['timestamp'].value_counts()
            duplicate_timestamps = (timestamp_counts > 1).sum()
            if duplicate_timestamps > 0:
                issues['duplicate_timestamps'] = duplicate_timestamps

        return issues

    def _check_data_continuity(self, df: pd.DataFrame) -> Dict[str, int]:
        """檢查數據連續性"""
        issues = {}

        if 'timestamp' not in df.columns or len(df) < 2:
            return issues

        # 計算時間間隔
        time_diffs = df['timestamp'].diff().dt.total_seconds()

        # 檢查是否有過大的時間間隔（可能表示數據丟失）
        if len(time_diffs) > 0:
            # 假設正常間隔應該不大於預期的最大間隔（例如1小時數據的最大間隔不應超過2小時）
            expected_intervals = self._infer_expected_intervals(df)
            if expected_intervals:
                max_expected_interval = expected_intervals * 2  # 允許一些彈性
                large_gaps = (time_diffs > max_expected_interval).sum()
                if large_gaps > 0:
                    issues['large_time_gaps'] = large_gaps

        return issues

    def _infer_expected_intervals(self, df: pd.DataFrame) -> Optional[float]:
        """推斷預期的時間間隔（秒）"""
        if len(df) < 2:
            return None

        time_diffs = df['timestamp'].diff().dt.total_seconds()
        # 取眾數作為預期間隔
        mode_diff = time_diffs.mode()

        if len(mode_diff) > 0:
            return float(mode_diff.iloc[0])

        return None

    def calculate_data_quality_score(self, df: pd.DataFrame) -> DataQualityMetrics:
        """
        計算數據質量指標

        Args:
            df: 要評估的DataFrame

        Returns:
            數據質量指標
        """
        if df.empty:
            return DataQualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        # 完整性：檢查缺失值的比例
        total_values = df.size
        missing_values = df.isna().sum().sum()
        completeness = 1.0 - (missing_values / total_values) if total_values > 0 else 0.0

        # 準確性：基於價格邏輯檢查
        price_logic_score = self._calculate_price_logic_score(df)

        # 一致性：檢查數據類型一致性和格式一致性
        consistency_score = self._calculate_consistency_score(df)

        # 及時性：檢查最新數據的時間（這裡簡化處理）
        timeliness_score = self._calculate_timeliness_score(df)

        # 綜合得分（加權平均）
        overall_score = (
            completeness * 0.3 +
            price_logic_score * 0.3 +
            consistency_score * 0.2 +
            timeliness_score * 0.2
        )

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=price_logic_score,
            consistency=consistency_score,
            timeliness=timeliness_score,
            overall_score=overall_score
        )

    def _calculate_price_logic_score(self, df: pd.DataFrame) -> float:
        """計算價格邏輯得分"""
        if df.empty:
            return 0.0

        score = 1.0

        # 檢查價格關係
        if 'high' in df.columns and 'low' in df.columns:
            high_low_issues = (df['high'] < df['low']).sum()
            score -= (high_low_issues / len(df)) * 0.5

        if 'high' in df.columns and 'close' in df.columns:
            high_close_issues = (df['high'] < df['close']).sum()
            score -= (high_close_issues / len(df)) * 0.3

        if 'low' in df.columns and 'close' in df.columns:
            low_close_issues = (df['low'] > df['close']).sum()
            score -= (low_close_issues / len(df)) * 0.3

        return max(0.0, score)

    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """計算一致性得分"""
        if df.empty:
            return 0.0

        score = 1.0

        # 檢查數據類型一致性
        for col in df.columns:
            if df[col].dtype == 'object':
                # 檢查字符串格式一致性（簡化處理）
                pass

        # 檢查數值範圍一致性（檢查是否有極端變化）
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and len(df[col].dropna()) > 1:
                returns = df[col].pct_change().dropna()
                extreme_changes = (abs(returns) > 0.5).sum()  # 超過50%的變化視為極端
                if extreme_changes > 0:
                    score -= (extreme_changes / len(returns)) * 0.1

        return max(0.0, score)

    def _calculate_timeliness_score(self, df: pd.DataFrame) -> float:
        """計算及時性得分"""
        if df.empty or 'timestamp' not in df.columns:
            return 0.0

        try:
            latest_timestamp = df['timestamp'].max()
            now = pd.Timestamp.now()

            # 計算時間差（小時）
            hours_diff = (now - latest_timestamp).total_seconds() / 3600

            # 簡單的及時性評分：24小時內為滿分，超過72小時為0分
            if hours_diff <= 24:
                return 1.0
            elif hours_diff <= 72:
                return 1.0 - ((hours_diff - 24) / 48)
            else:
                return 0.0

        except Exception:
            return 0.5  # 無法計算時給中等分數

    def generate_validation_report(self, df: pd.DataFrame, symbol: str = None) -> str:
        """
        生成數據驗證報告

        Args:
            df: 要驗證的DataFrame
            symbol: 交易對符號

        Returns:
            格式化的報告字符串
        """
        result = self.validate_ohlcv_data(df, symbol)
        quality = self.calculate_data_quality_score(df)

        report = []
        report.append("=" * 60)
        report.append("數據驗證報告")
        report.append("=" * 60)

        if symbol:
            report.append(f"交易對: {symbol}")

        report.append(f"總記錄數: {result.total_records}")
        report.append(f"有效記錄數: {result.valid_records}")
        report.append(f"驗證結果: {'通過' if result.is_valid else '失敗'}")
        report.append("")

        # 數據質量指標
        report.append("數據質量指標:")
        report.append(f"  完整性: {quality.completeness:.2%}")
        report.append(f"  準確性: {quality.accuracy:.2%}")
        report.append(f"  一致性: {quality.consistency:.2%}")
        report.append(f"  及時性: {quality.timeliness:.2%}")
        report.append(f"  綜合得分: {quality.overall_score:.2%}")
        report.append("")

        # 問題統計
        if result.issues_found:
            report.append("發現的問題:")
            for issue, count in result.issues_found.items():
                report.append(f"  {issue}: {count}")
            report.append("")

        # 警告
        if result.warnings:
            report.append("警告:")
            for warning in result.warnings:
                report.append(f"  • {warning}")
            report.append("")

        # 錯誤
        if result.errors:
            report.append("錯誤:")
            for error in result.errors:
                report.append(f"  ✗ {error}")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)


# 使用範例
if __name__ == "__main__":
    # 創建驗證器
    validator = DataValidator(strict_mode=False)

    # 創建範例數據進行測試
    sample_data = {
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }

    df = pd.DataFrame(sample_data)

    # 執行驗證
    result = validator.validate_ohlcv_data(df, "BTC/USDT")

    print("驗證結果:")
    print(f"有效: {result.is_valid}")
    print(f"總記錄: {result.total_records}")
    print(f"有效記錄: {result.valid_records}")
    print(f"發現問題: {result.issues_found}")

    # 生成報告
    report = validator.generate_validation_report(df, "BTC/USDT")
    print("\n詳細報告:")
    print(report)
