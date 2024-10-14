export interface AntiCheatResult {
  fileName: string;
  player: string;
  status: 'Normal' | 'Suspicious' | 'Cheating';
  cheatProbability: number;
}