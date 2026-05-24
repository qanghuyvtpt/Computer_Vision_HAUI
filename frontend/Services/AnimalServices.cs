using System.Net.Http.Json;
using System.Text.Json.Serialization;
using btlon.Models;
using Microsoft.Extensions.Options;

namespace btlon.Services
{
    public class ModelApiOptions
    {
        public const string SectionName = "ModelApi";
        public string BaseUrl { get; set; } = "http://localhost:8000";
        public bool UseMock { get; set; } = true;
    }

    /* ── HTTP client → Python FastAPI ───────────────────── */
    public class HttpAnimalClassificationService : IAnimalClassificationService
    {
        private readonly HttpClient _http;
        private readonly ILogger<HttpAnimalClassificationService> _log;

        public HttpAnimalClassificationService(
            HttpClient http,
            IOptions<ModelApiOptions> options,
            ILogger<HttpAnimalClassificationService> log)
        {
            _http = http;
            _log = log;
            var baseUrl = options.Value.BaseUrl.TrimEnd('/');
            if (_http.BaseAddress == null)
                _http.BaseAddress = new Uri(baseUrl + "/");
            _http.Timeout = TimeSpan.FromMinutes(5);
        }

        public async Task<ClassifyResultDto> ClassifyAsync(string imagePath)
        {
            await using var fs = File.OpenRead(imagePath);
            var fileName = Path.GetFileName(imagePath);
            var streamContent = new StreamContent(fs);
            streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(
                GetImageContentType(fileName));

            using var form = new MultipartFormDataContent();
            form.Add(streamContent, "file", fileName);

            HttpResponseMessage response;
            try
            {
                response = await _http.PostAsync("api/classify", form);
            }
            catch (HttpRequestException ex)
            {
                _log.LogError(ex, "Không kết nối được API model tại {Base}", _http.BaseAddress);
                throw new InvalidOperationException(
                    "Không kết nối được backend AI. Hãy chạy: uvicorn backend.app:app --port 8000",
                    ex);
            }

            if (!response.IsSuccessStatusCode)
            {
                var body = await response.Content.ReadAsStringAsync();
                _log.LogWarning("API lỗi {Code}: {Body}", response.StatusCode, body);
                throw new InvalidOperationException(
                    $"Backend AI trả lỗi {(int)response.StatusCode}: {body}");
            }

            var api = await response.Content.ReadFromJsonAsync<ClassifyApiResponse>()
                      ?? throw new InvalidOperationException("Phản hồi API rỗng.");

            return MapToDto(api);
        }

        private static string GetImageContentType(string fileName)
        {
            return Path.GetExtension(fileName).ToLowerInvariant() switch
            {
                ".jpg" or ".jpeg" => "image/jpeg",
                ".png" => "image/png",
                ".gif" => "image/gif",
                ".webp" => "image/webp",
                ".bmp" => "image/bmp",
                _ => "application/octet-stream"
            };
        }

        private static ClassifyResultDto MapToDto(ClassifyApiResponse api)
        {
            var diet = Enum.TryParse<DietType>(api.DietType, true, out var dt)
                ? dt
                : DietType.Herbivore;

            return new ClassifyResultDto
            {
                Label = api.Label,
                DietType = diet,
                Confidence = api.Confidence,
                HerbivorePct = api.HerbivorePct,
                CarnivorePct = api.CarnivorePct,
                OmnivorePct = api.OmnivorePct,
                Traits = api.Traits ?? new List<string>(),
                Boxes = api.Boxes?.Select(b => new DetectionBox
                {
                    X = b.X,
                    Y = b.Y,
                    Width = b.Width,
                    Height = b.Height,
                    DietType = Enum.TryParse<DietType>(b.DietType, true, out var boxDt)
                        ? boxDt
                        : DietType.Herbivore,
                    Label = b.Label,
                    Confidence = b.Confidence
                }).ToList() ?? new List<DetectionBox>()
            };
        }
    }

    internal sealed class ClassifyApiResponse
    {
        public string Label { get; set; } = "";
        [JsonPropertyName("dietType")]
        public string DietType { get; set; } = "";
        public double Confidence { get; set; }
        [JsonPropertyName("herbivorePct")]
        public double HerbivorePct { get; set; }
        [JsonPropertyName("carnivorePct")]
        public double CarnivorePct { get; set; }
        [JsonPropertyName("omnivorePct")]
        public double OmnivorePct { get; set; }
        public List<string>? Traits { get; set; }
        public List<ClassifyApiBox>? Boxes { get; set; }
    }

    internal sealed class ClassifyApiBox
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
        [JsonPropertyName("dietType")]
        public string DietType { get; set; } = "";
        public string Label { get; set; } = "";
        public double Confidence { get; set; }
    }

    /* ── Interface ──────────────────────────────────────── */
    public interface IAnimalClassificationService
    {
        Task<ClassifyResultDto> ClassifyAsync(string imagePath);
    }

    public class ClassifyResultDto
    {
        public string Label { get; set; } = "";
        public DietType DietType { get; set; }
        public double Confidence { get; set; }
        public double HerbivorePct { get; set; }
        public double CarnivorePct { get; set; }
        public double OmnivorePct { get; set; }
        // Bỏ AnimalName theo yêu cầu
        public List<string> Traits { get; set; } = new();
        public List<DetectionBox> Boxes { get; set; } = new();
    }

    /* ── Mock Service — 3 loại, nhiều box ───────────────── */
    public class MockAnimalClassificationService : IAnimalClassificationService
    {
        private readonly Random _rng = new();

        // Pool dữ liệu — KHÔNG có tên động vật
        private static readonly (DietType type, double conf, double h, double c, double o, string[] traits)[] _pool =
        [
            (DietType.Herbivore, .91, .91, .04, .05, ["Ăn cỏ","Dạ dày 4 ngăn","Nhai lại","Móng guốc"]),
            (DietType.Herbivore, .88, .88, .05, .07, ["Ăn cây bụi","Chạy nhanh","Sừng","Mắt bên hông"]),
            (DietType.Herbivore, .95, .95, .02, .03, ["Ăn rau củ","Tai dài","Răng cửa to","Sinh sản nhanh"]),
            (DietType.Herbivore, .93, .93, .03, .04, ["Ăn cỏ","Móng guốc","Ruột dài","Chạy nhanh"]),
            (DietType.Carnivore, .97, .02, .97, .01, ["Nanh sắc","Móng vuốt","Mắt hướng trước","Săn theo bầy"]),
            (DietType.Carnivore, .94, .03, .94, .03, ["Săn theo bầy","Nanh dài","Lãnh thổ cao"]),
            (DietType.Carnivore, .96, .02, .96, .02, ["Mỏ cong sắc","Móng vuốt mạnh","Thị lực x8"]),
            (DietType.Carnivore, .98, .01, .98, .01, ["Hàm lực cực mạnh","Bán thủy sinh","Phục kích"]),
            (DietType.Omnivore,  .82, .20, .15, .82, ["Ăn cả thịt & quả","Ngủ đông","Móng vuốt"]),
            (DietType.Omnivore,  .85, .25, .18, .85, ["Ăn mọi thứ","Mõm dài","Thích nghi cao"]),
            (DietType.Omnivore,  .79, .30, .12, .79, ["Thông minh cao","Ăn hạt & sâu","Nhớ mặt người"]),
        ];

        public Task<ClassifyResultDto> ClassifyAsync(string imagePath)
        {
            Thread.Sleep(_rng.Next(400, 900));

            // Sinh 1–4 box ngẫu nhiên
            int boxCount = _rng.Next(1, 5);
            var boxes = new List<DetectionBox>();

            for (int i = 0; i < boxCount; i++)
            {
                var pick = _pool[_rng.Next(_pool.Length)];
                // Tọa độ normalized 0–1, không chồng lên nhau nhiều
                double x = Math.Round(_rng.NextDouble() * 0.6, 3);
                double y = Math.Round(_rng.NextDouble() * 0.6, 3);
                double w = Math.Round(0.15 + _rng.NextDouble() * 0.25, 3);
                double h = Math.Round(0.15 + _rng.NextDouble() * 0.25, 3);
                // Clamp
                if (x + w > 1) w = 1 - x;
                if (y + h > 1) h = 1 - y;

                boxes.Add(new DetectionBox
                {
                    X = x,
                    Y = y,
                    Width = w,
                    Height = h,
                    DietType = pick.type,
                    Label = pick.type.ToString().ToLower(),
                    Confidence = Math.Round(pick.conf - _rng.NextDouble() * 0.1 + _rng.NextDouble() * 0.05, 4)
                });
            }

            // Kết quả tổng = box đầu tiên
            var main = boxes[0];
            var mainPick = _pool.First(p => p.type == main.DietType);

            return Task.FromResult(new ClassifyResultDto
            {
                Label = main.DietType == DietType.Carnivore ? "Động vật ăn thịt"
                             : main.DietType == DietType.Omnivore ? "Động vật tạp ăn"
                             : "Động vật ăn cỏ",
                DietType = main.DietType,
                Confidence = main.Confidence,
                HerbivorePct = mainPick.h,
                CarnivorePct = mainPick.c,
                OmnivorePct = mainPick.o,
                Traits = mainPick.traits.ToList(),
                Boxes = boxes
            });
        }
    }

    /* ── History Service ─────────────────────────────────── */
    public interface IHistoryService
    {
        void Add(ClassifyHistory item);
        List<ClassifyHistory> GetAll();
        ClassifyHistory? GetById(int id);
        void Delete(int id);
        void Clear();
        StatsViewModel GetStats();
    }

    public class InMemoryHistoryService : IHistoryService
    {
        private static readonly List<ClassifyHistory> _store = new();
        private static int _nextId = 1;
        private static readonly object _lock = new();

        public void Add(ClassifyHistory item)
        {
            lock (_lock) { item.Id = _nextId++; _store.Insert(0, item); }
        }
        public List<ClassifyHistory> GetAll()
        {
            lock (_lock) { return _store.ToList(); }
        }
        public ClassifyHistory? GetById(int id)
        {
            lock (_lock) { return _store.FirstOrDefault(x => x.Id == id); }
        }
        public void Delete(int id)
        {
            lock (_lock) { var x = _store.FirstOrDefault(i => i.Id == id); if (x != null) _store.Remove(x); }
        }
        public void Clear()
        {
            lock (_lock) { _store.Clear(); }
        }
        public StatsViewModel GetStats()
        {
            lock (_lock)
            {
                var all = _store.ToList();
                return new StatsViewModel
                {
                    Total = all.Count,
                    Herbivore = all.Count(x => x.DietType == DietType.Herbivore),
                    Carnivore = all.Count(x => x.DietType == DietType.Carnivore),
                    Omnivore = all.Count(x => x.DietType == DietType.Omnivore),
                    AvgConf = all.Any() ? Math.Round(all.Average(x => x.Confidence) * 100, 1) : 0,
                    Recent = all.Take(5).ToList()
                };
            }
        }
    }
}