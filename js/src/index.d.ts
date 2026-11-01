export interface AksharaUnit {
  akshara: string;
  weight: 'G' | 'L' | 'C';
}

export interface G2PResult {
  inputIds: number[];
  attentionMask: number[];
  ipa: string;
  ipaIds: number[];
  lang: 'sa' | 'hi' | 'kn' | 'ta' | 'te' | 'ml';
  langId: number;
  script: string;
  normalizedDevanagari: string;
  tokens: string[];
  aksharas: AksharaUnit[];
}

export interface StreamChunk {
  pcm: Float32Array;
  wav: ArrayBuffer;
  sampleRate: number;
  index: number;
  clauseIndex: number;
  clauseCount: number;
  isFinal: boolean;
  durationSec: number;
  latencyMs: number;
  ttfaMs: number;
  lang: string;
  script: string;
  tier: 'neural_onnx' | 'klatt_zero_dep';
  aksharas: AksharaUnit[];
  durations: number[];
}

export interface SynthesisMeta extends G2PResult {
  tier: 'neural_onnx' | 'klatt_zero_dep';
  sampleRate: number;
  durationSec: number;
  latencyMs: number;
  ttfaMs: number;
  chunks: number;
  rtf: number;
  durations?: number[];
}

export type VoiceBankSource = string | ArrayBuffer | Uint8Array | Float32Array;

export interface NinadaOptions {
  lang?: string;
  model?: string | ArrayBuffer | Uint8Array;
  voice?: VoiceBankSource;
  voiceBank?: VoiceBankSource;
  voices?: Record<string, VoiceBankSource>;
  encoderModel?: string | ArrayBuffer | Uint8Array;
  vocoderModel?: string | ArrayBuffer | Uint8Array;
  session?: unknown;
  encoderSession?: unknown;
  vocoderSession?: unknown;
  ort?: unknown;
  providers?: string[];
  cache?: boolean;
  rate?: number;
  noise?: number;
  noiseDuration?: number;
  noise_duration?: number;
}

export interface SpeakOptions {
  lang?: string;
  rate?: number;
  noise?: number;
  noiseDuration?: number;
  noise_duration?: number;
  tier?: 'auto' | 'neural' | 'klatt';
  firstChunkFrames?: number;
  chunkFrames?: number;
  haloFrames?: number;
  maxClauseChars?: number;
  fastFirstClause?: boolean;
  signal?: AbortSignal;
}

export interface NinadaVoice {
  readonly lang: string;
  readonly isNeural: boolean;
  readonly isStreamingNeural: boolean;
  setVoiceBank(lang: string, bank: ArrayBuffer | Uint8Array | Float32Array): this;
  dispose(): Promise<void>;
  on(event: 'chunk' | 'done', handler: (payload: unknown) => void): this;
  inspect(text: string, opts?: SpeakOptions): G2PResult;
  stream(text: string, opts?: SpeakOptions): AsyncGenerator<StreamChunk, void, unknown>;
  synthesize(text: string, opts?: SpeakOptions): Promise<{ pcm: Float32Array; wav: ArrayBuffer; meta: SynthesisMeta }>;
  speak(text: string, opts?: SpeakOptions): Promise<ArrayBuffer>;
}

export default function ninada(input?: string | ArrayBuffer | Uint8Array | NinadaOptions, opts?: NinadaOptions): Promise<NinadaVoice>;
export { ninada };
export const SAMPLE_RATE: number;
export const HOP_LENGTH: number;
export const VOCAB_TOKENS: string[];
export const TOKEN_TO_ID: Record<string, number>;
export const IPA_VOCAB: Record<string, number>;
export const IPA_ID_TO_TOKEN: Record<number, string>;
export const RETROFLEX_LATERAL_ID: number;
export const BLANK_ID: number;
export const LANGUAGE_IDS: Record<string, number>;
export function splitProsodicClauses(text: string, maxChars?: number): string[];
export function detectScriptAndLang(text: string, defaultLang?: string): [string, string];
export function iastToDevanagari(text: string): string;
export function brahmicToDevanagari(text: string): string;
export function applyPhonologicalRules(devText: string, lang?: string): string;
export function segmentAksharas(devText: string): AksharaUnit[];
export function textToIpa(text: string, lang?: string, normalizedDevFallback?: string): string;
export function ipaToIds(ipa: string, maxTokens?: number): number[];
export function textToIds(text: string, opts?: { lang?: string; addBlank?: boolean; minLen?: number }): G2PResult;
export function synthesizeKlattPcm(text: string, opts?: { lang?: string; rate?: number; f0Base?: number }): { pcm: Float32Array; meta: G2PResult & { tier: string } };
export function encodeWavBuffer(pcm: Float32Array, sr?: number): ArrayBuffer;
