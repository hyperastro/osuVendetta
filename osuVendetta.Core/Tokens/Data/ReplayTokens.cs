﻿using osuVendetta.Core.AntiCheat;

namespace osuVendetta.Core.Replays.Data;

/// <summary>
/// Input tokens for the <see cref="IAntiCheatModel"/>
/// </summary>
public class ReplayTokens
{
    public required string ReplayName { get; set; }
    public required float[,,] Tokens { get; set; }
}