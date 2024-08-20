using Microsoft.AspNetCore.Mvc;

namespace osuvendetta.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
public class FileController : Controller
{
    // /api/File?file=replay.osr
    [HttpGet]
    public IActionResult Index(string? file = null)
    {
        if (string.IsNullOrEmpty(file))
            return NotFound();

        string path = Path.Combine("./wwwroot/onnx/", file);

        if (!System.IO.File.Exists(path))
            return NotFound($"File not found: {file ?? string.Empty}");

        byte[] fileData = System.IO.File.ReadAllBytes(path);
        return File(fileData, "application/octet-stream", file);
    }
}
