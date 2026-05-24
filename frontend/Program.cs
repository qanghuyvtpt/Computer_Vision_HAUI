using btlon.Services;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllersWithViews();

builder.Services.Configure<ModelApiOptions>(
    builder.Configuration.GetSection(ModelApiOptions.SectionName));

var modelApi = builder.Configuration
    .GetSection(ModelApiOptions.SectionName)
    .Get<ModelApiOptions>() ?? new ModelApiOptions();

if (modelApi.UseMock)
{
    builder.Services.AddScoped<IAnimalClassificationService, MockAnimalClassificationService>();
}
else
{
    builder.Services.AddHttpClient<IAnimalClassificationService, HttpAnimalClassificationService>(
        (sp, client) =>
        {
            var opts = sp.GetRequiredService<Microsoft.Extensions.Options.IOptions<ModelApiOptions>>().Value;
            client.BaseAddress = new Uri(opts.BaseUrl.TrimEnd('/') + "/");
            client.Timeout = TimeSpan.FromMinutes(5);
        });
}

// History lưu in-memory (Singleton để giữ dữ liệu qua request)
builder.Services.AddSingleton<IHistoryService, InMemoryHistoryService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();