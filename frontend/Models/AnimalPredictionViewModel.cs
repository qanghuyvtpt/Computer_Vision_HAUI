using Microsoft.AspNetCore.Http;

namespace btlon.Models
{
    public enum DietType { Herbivore, Carnivore, Omnivore }

    // ── Bounding box 1 con vật ─────────────────────────────
    public class DetectionBox
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
        public DietType DietType { get; set; }
        public string Label { get; set; } = "";
        public double Confidence { get; set; }

        // helpers
        public string TypeClass => DietType switch
        {
            DietType.Carnivore => "carnivore",
            DietType.Omnivore => "omnivore",
            _ => "herbivore"
        };
        public string TypeEmoji => DietType switch
        {
            DietType.Carnivore => "🥩",
            DietType.Omnivore => "🍽️",
            _ => "🌿"
        };
        public string LabelVi => DietType switch
        {
            DietType.Carnivore => "Ăn thịt",
            DietType.Omnivore => "Tạp ăn",
            _ => "Ăn cỏ"
        };
        public string ConfidencePercent => $"{Confidence * 100:F1}%";
    }

    // ── ViewModel trang Index ──────────────────────────────
    public class AnimalPredictionViewModel
    {
        public IFormFile? ImageFile { get; set; }
        public string? UploadedImagePath { get; set; }
        public string? PredictedLabel { get; set; }
        public DietType DietType { get; set; }
        public double Confidence { get; set; }
        public double HerbivorePct { get; set; }
        public double CarnivorePct { get; set; }
        public double OmnivorePct { get; set; }
        public List<string> Traits { get; set; } = new();
        public List<DetectionBox> DetectionBoxes { get; set; } = new();

        public string ConfidencePercent => $"{Confidence * 100:F1}%";
        public bool HasResult => !string.IsNullOrEmpty(PredictedLabel);

        public string TypeClass => DietType switch
        {
            DietType.Carnivore => "carnivore",
            DietType.Omnivore => "omnivore",
            _ => "herbivore"
        };
        public string TypeIcon => DietType switch
        {
            DietType.Carnivore => "🥩",
            DietType.Omnivore => "🍽️",
            _ => "🌿"
        };
        public string TypeEmoji => TypeIcon;
    }

    // ── Lịch sử — lưu ĐỦ từng box ────────────────────────
    public class ClassifyHistory
    {
        public int Id { get; set; }
        public string ImagePath { get; set; } = "";
        public string PredictedLabel { get; set; } = "";
        public DietType DietType { get; set; }
        public double Confidence { get; set; }
        public int BoxCount { get; set; }
        // ★ LƯU TOÀN BỘ BOXES để History hiện đủ
        public List<DetectionBox> Boxes { get; set; } = new();
        public DateTime CreatedAt { get; set; } = DateTime.Now;

        public string TypeClass => DietType switch
        {
            DietType.Carnivore => "carnivore",
            DietType.Omnivore => "omnivore",
            _ => "herbivore"
        };
        public string TypeIcon => DietType switch
        {
            DietType.Carnivore => "🥩",
            DietType.Omnivore => "🍽️",
            _ => "🌿"
        };
        public string ConfidencePercent => $"{Confidence * 100:F1}%";
    }

    // ── Stats ──────────────────────────────────────────────
    public class StatsViewModel
    {
        public int Total { get; set; }
        public int Herbivore { get; set; }
        public int Carnivore { get; set; }
        public int Omnivore { get; set; }
        public double AvgConf { get; set; }
        public List<ClassifyHistory> Recent { get; set; } = new();

        public double HerbivorePct => Total > 0 ? Math.Round((double)Herbivore / Total * 100, 1) : 0;
        public double CarnivorePct => Total > 0 ? Math.Round((double)Carnivore / Total * 100, 1) : 0;
        public double OmnivorePct => Total > 0 ? Math.Round((double)Omnivore / Total * 100, 1) : 0;
    }

    public class ErrorViewModel
    {
        public string? RequestId { get; set; }
        public bool ShowRequestId => !string.IsNullOrEmpty(RequestId);
    }
}