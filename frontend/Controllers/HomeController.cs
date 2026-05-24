using btlon.Models;
using btlon.Services;
using Microsoft.AspNetCore.Mvc;

namespace btlon.Controllers
{
    public class HomeController : Controller
    {
        private readonly IAnimalClassificationService _svc;
        private readonly IHistoryService _hist;
        private readonly IWebHostEnvironment _env;
        private readonly ILogger<HomeController> _log;
        private readonly IConfiguration _config;

        private static readonly HashSet<string> _allowed =
            new(StringComparer.OrdinalIgnoreCase)
            { ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp" };

        public HomeController(
            IAnimalClassificationService svc,
            IHistoryService hist,
            IWebHostEnvironment env,
            ILogger<HomeController> log,
            IConfiguration config)
        {
            _svc = svc;
            _hist = hist;
            _env = env;
            _log = log;
            _config = config;
        }

        private void SetApiModeViewBag()
            => ViewBag.UseMock = _config.GetValue<bool>("ModelApi:UseMock", true);

        [HttpGet]
        public IActionResult Index()
        {
            SetApiModeViewBag();
            return View(new AnimalPredictionViewModel());
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        [RequestSizeLimit(10 * 1024 * 1024)]
        public async Task<IActionResult> Index(AnimalPredictionViewModel model)
        {
            SetApiModeViewBag();

            if (model.ImageFile is null || model.ImageFile.Length == 0)
            {
                ModelState.AddModelError("ImageFile", "Vui lòng chọn một file ảnh.");
                return View(model);
            }
            var ext = Path.GetExtension(model.ImageFile.FileName).ToLower();
            if (!_allowed.Contains(ext))
            {
                ModelState.AddModelError("ImageFile", "Chỉ chấp nhận: JPG, PNG, GIF, WEBP, BMP.");
                return View(model);
            }

            var uploadsDir = Path.Combine(_env.WebRootPath, "uploads");
            Directory.CreateDirectory(uploadsDir);
            var fileName = $"{Guid.NewGuid()}{ext}";
            var filePath = Path.Combine(uploadsDir, fileName);
            await using (var fs = new FileStream(filePath, FileMode.Create))
                await model.ImageFile.CopyToAsync(fs);
            _log.LogInformation("Saved: {F}", filePath);

            try
            {
                var result = await _svc.ClassifyAsync(filePath);
                ApplyClassifyResult(model, result, fileName);
            }
            catch (InvalidOperationException ex)
            {
                ModelState.AddModelError("", ex.Message);
                return View(model);
            }

            return View(model);
        }

        private void ApplyClassifyResult(
            AnimalPredictionViewModel model,
            ClassifyResultDto result,
            string fileName)
        {
            model.PredictedLabel = result.Label;
            model.DietType = result.DietType;
            model.Confidence = result.Confidence;
            model.HerbivorePct = result.HerbivorePct;
            model.CarnivorePct = result.CarnivorePct;
            model.OmnivorePct = result.OmnivorePct;
            model.Traits = result.Traits;
            model.DetectionBoxes = result.Boxes;
            model.UploadedImagePath = $"/uploads/{fileName}";

            _hist.Add(new ClassifyHistory
            {
                ImagePath = model.UploadedImagePath,
                PredictedLabel = result.Label,
                DietType = result.DietType,
                Confidence = result.Confidence,
                BoxCount = result.Boxes.Count,
                Boxes = result.Boxes,
                CreatedAt = DateTime.Now
            });
        }

        // ── History — hỗ trợ xem chi tiết từng lần ──────────
        public IActionResult History(string filter = "all")
        {
            var list = _hist.GetAll();
            if (filter == "herb") list = list.Where(x => x.DietType == DietType.Herbivore).ToList();
            else if (filter == "carn") list = list.Where(x => x.DietType == DietType.Carnivore).ToList();
            else if (filter == "omni") list = list.Where(x => x.DietType == DietType.Omnivore).ToList();
            ViewBag.Filter = filter;
            return View(list);
        }

        // ★ Action mới: trả JSON detail 1 lần phân tích
        [HttpGet]
        public IActionResult HistoryDetail(int id)
        {
            var item = _hist.GetById(id);
            if (item == null) return NotFound();
            return Json(new
            {
                id = item.Id,
                imagePath = item.ImagePath,
                label = item.PredictedLabel,
                confidence = item.ConfidencePercent,
                boxCount = item.BoxCount,
                createdAt = item.CreatedAt.ToString("dd/MM/yyyy HH:mm:ss"),
                boxes = item.Boxes.Select((b, i) => new
                {
                    idx = i + 1,
                    labelVi = b.LabelVi,
                    typeClass = b.TypeClass,
                    typeEmoji = b.TypeEmoji,
                    confidence = b.ConfidencePercent,
                    confRaw = b.Confidence,
                    x = b.X,
                    y = b.Y,
                    w = b.Width,
                    h = b.Height
                })
            });
        }

        [HttpPost]
        public IActionResult Delete(int id)
        {
            _hist.Delete(id);
            return RedirectToAction(nameof(History));
        }

        [HttpPost]
        public IActionResult ClearHistory()
        {
            _hist.Clear();
            return RedirectToAction(nameof(History));
        }

        public IActionResult Stats() => View(_hist.GetStats());

        [HttpGet]
        public IActionResult StatsJson()
        {
            var s = _hist.GetStats();
            return Json(new
            {
                total = s.Total,
                herbivore = s.Herbivore,
                carnivore = s.Carnivore,
                omnivore = s.Omnivore,
                herbPct = s.HerbivorePct,
                carnPct = s.CarnivorePct,
                omniPct = s.OmnivorePct,
                avgConf = s.AvgConf
            });
        }

        public IActionResult Info() => View();

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
            => View(new ErrorViewModel
            { RequestId = System.Diagnostics.Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}