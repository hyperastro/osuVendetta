using Microsoft.AspNetCore.Mvc;

namespace osuVendetta.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
public class FileController : Controller
{
    // /api/File?file=replay.osr
    [HttpGet]
    public IActionResult Index(string? version = "192x2.onnx")
    {
        if (string.IsNullOrEmpty(version))
            return NotFound();

        string path = Path.Combine("./wwwroot/onnx/", version);

        if (!System.IO.File.Exists(path))
            return NotFound($"File not found: {version ?? string.Empty}");

        byte[] fileData = System.IO.File.ReadAllBytes(path);
        return File(fileData, "application/octet-stream", version);
    }
}
