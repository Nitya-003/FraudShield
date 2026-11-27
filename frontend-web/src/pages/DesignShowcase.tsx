import React, { useState } from 'react';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Tabs, TabsContent } from '../components/ui/Tabs';
import { Dialog } from '../components/ui/Dialog';
import { Drawer } from '../components/ui/Drawer';
import { useToast } from '../components/ui/Toast';
import {
  Palette,
  Type,
  Component as ComponentIcon,
  Search,
  Sparkles,
  Shield,
  Layers,
  CheckCircle2,
} from 'lucide-react';

export interface DesignShowcaseProps {
  isDark?: boolean;
  onOpenCommandPalette: () => void;
}

export const DesignShowcase: React.FC<DesignShowcaseProps> = ({ onOpenCommandPalette }) => {
  const [activeTab, setActiveTab] = useState('tokens');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { toast } = useToast();

  const showcaseTabs = [
    { id: 'tokens', label: 'Color Tokens & Contrast', icon: Palette },
    { id: 'typography', label: 'Typography Scale', icon: Type },
    { id: 'primitives', label: 'Restyled Primitives', icon: ComponentIcon },
  ];

  return (
    <div className="space-y-8 pb-16 font-sans">
      {/* Dev-Only Guard Header */}
      <div className="border-b border-[#E3E8EE] dark:border-[#1E2630] pb-5">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-sans font-semibold px-2 py-0.5 rounded-sm bg-accent/10 text-accent border border-accent/20">
            Dev-Only Route
          </span>
          <span className="text-xs font-mono text-[#52606D] dark:text-[#8A96A3]">
            Excluded from production builds
          </span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-normal text-text">
              Security Control Room Visual Identity
            </h1>
            <p className="text-xs text-[#52606D] dark:text-[#8A96A3] mt-1">
              Design tokens, grotesque typography, Radix/cmdk/sonner/vaul primitives, and contrast audits.
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={onOpenCommandPalette}>
            <Search className="w-3.5 h-3.5" strokeWidth={1.5} />
            <span>Open Command Palette (Ctrl+K)</span>
          </Button>
        </div>
      </div>

      {/* Tabs Navigation */}
      <Tabs tabs={showcaseTabs} value={activeTab} onValueChange={setActiveTab}>
        {/* Tab 1: Color Tokens & WCAG Contrast */}
        <TabsContent value="tokens" className="pt-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Dark Theme Palette (Primary Control Room) */}
            <div className="p-5 rounded-lg border border-[#1E2630] bg-[#11171E] space-y-4">
              <div className="flex items-center justify-between border-b border-[#1E2630] pb-3">
                <div>
                  <h3 className="font-semibold text-sm text-[#E8EDF2]">Dark Theme (Primary Control Room)</h3>
                  <p className="text-xs text-[#8A96A3]">Direction: Calm, high-density, precise</p>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#161F28] border border-[#1E2630] text-[#2DD4A0]">
                  Default Theme
                </span>
              </div>

              <div className="space-y-2.5 text-xs font-mono">
                <div className="flex items-center justify-between p-2.5 rounded bg-[#0B0F14] border border-[#1E2630]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#0B0F14] border border-[#1E2630]" />
                    <span className="text-[#E8EDF2]">Background</span>
                  </div>
                  <span className="text-[#8A96A3]">#0B0F14</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-[#11171E] border border-[#1E2630]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#11171E] border border-[#2A384A]" />
                    <span className="text-[#E8EDF2]">Surface / Card</span>
                  </div>
                  <span className="text-[#8A96A3]">#11171E</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-[#161F28] border border-[#1E2630]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#1E2630]" />
                    <span className="text-[#E8EDF2]">Border Rule</span>
                  </div>
                  <span className="text-[#8A96A3]">#1E2630</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-[#11171E] border border-[#1E2630]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#2DD4A0]" />
                    <span className="text-[#E8EDF2]">Brand Accent (Emerald)</span>
                  </div>
                  <span className="text-[#2DD4A0]">#2DD4A0</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-[#11171E] border border-[#1E2630]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#F59E0B]" />
                    <span className="text-[#E8EDF2]">Risk: Medium (Amber)</span>
                  </div>
                  <span className="text-[#F59E0B]">#F59E0B</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-[#11171E] border border-[#1E2630]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#F43F5E]" />
                    <span className="text-[#E8EDF2]">Risk: High (Red)</span>
                  </div>
                  <span className="text-[#F43F5E]">#F43F5E</span>
                </div>
              </div>
            </div>

            {/* Light Theme Palette (Designed, Not Inverted) */}
            <div className="p-5 rounded-lg border border-[#E3E8EE] bg-white space-y-4">
              <div className="flex items-center justify-between border-b border-[#E3E8EE] pb-3">
                <div>
                  <h3 className="font-semibold text-sm text-[#0F172A]">Light Theme (Designed Precision)</h3>
                  <p className="text-xs text-[#52606D]">Direction: Crisp, high-contrast, zero eye fatigue</p>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 border border-[#E3E8EE] text-[#0E9F77]">
                  Designed Light
                </span>
              </div>

              <div className="space-y-2.5 text-xs font-mono">
                <div className="flex items-center justify-between p-2.5 rounded bg-[#F6F8FA] border border-[#E3E8EE]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#F6F8FA] border border-[#E3E8EE]" />
                    <span className="text-[#0F172A]">Background</span>
                  </div>
                  <span className="text-[#52606D]">#F6F8FA</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-white border border-[#E3E8EE]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-white border border-[#CBD5E1]" />
                    <span className="text-[#0F172A]">Surface / Card</span>
                  </div>
                  <span className="text-[#52606D]">#FFFFFF</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-slate-50 border border-[#E3E8EE]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#E3E8EE]" />
                    <span className="text-[#0F172A]">Border Rule</span>
                  </div>
                  <span className="text-[#52606D]">#E3E8EE</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-white border border-[#E3E8EE]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#0E9F77]" />
                    <span className="text-[#0F172A]">Brand Accent (Emerald)</span>
                  </div>
                  <span className="text-[#0E9F77]">#0E9F77</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-white border border-[#E3E8EE]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#D97706]" />
                    <span className="text-[#0F172A]">Risk: Medium (Amber)</span>
                  </div>
                  <span className="text-[#D97706]">#D97706</span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded bg-white border border-[#E3E8EE]">
                  <div className="flex items-center gap-2.5">
                    <div className="w-5 h-5 rounded bg-[#DC2626]" />
                    <span className="text-[#0F172A]">Risk: High (Red)</span>
                  </div>
                  <span className="text-[#DC2626]">#DC2626</span>
                </div>
              </div>
            </div>
          </div>

          {/* Contrast Verification Matrix (WCAG AA Compliance) */}
          <div className="p-5 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-3">
            <h3 className="font-semibold text-xs text-muted">
              WCAG 2.1 AA Contrast Ratios (Minimum 4.5:1 Requirement)
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-3 rounded border border-[#1E2630] bg-[#0B0F14]">
                <div className="text-[#8A96A3] text-xs">DARK BG : TEXT</div>
                <div className="text-sm font-semibold text-[#E8EDF2] mt-1">16.5 : 1</div>
                <div className="text-xs text-[#2DD4A0]">PASS (AAA)</div>
              </div>
              <div className="p-3 rounded border border-[#1E2630] bg-[#11171E]">
                <div className="text-[#8A96A3] text-xs">DARK SURFACE : MUTED</div>
                <div className="text-sm font-semibold text-[#8A96A3] mt-1">5.7 : 1</div>
                <div className="text-xs text-[#2DD4A0]">PASS (AA)</div>
              </div>
              <div className="p-3 rounded border border-[#E3E8EE] bg-[#F6F8FA]">
                <div className="text-[#52606D] text-xs">LIGHT BG : TEXT</div>
                <div className="text-sm font-semibold text-[#0F172A] mt-1">15.6 : 1</div>
                <div className="text-xs text-[#0E9F77]">PASS (AAA)</div>
              </div>
              <div className="p-3 rounded border border-[#E3E8EE] bg-white">
                <div className="text-[#52606D] text-xs">LIGHT SURFACE : MUTED</div>
                <div className="text-sm font-semibold text-[#52606D] mt-1">5.8 : 1</div>
                <div className="text-xs text-[#0E9F77]">PASS (AA)</div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Tab 2: Typography Scale */}
        <TabsContent value="typography" className="pt-6 space-y-6">
          <div className="p-5 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-5">
            <div>
              <h3 className="font-semibold text-sm text-[#0F172A] dark:text-[#E8EDF2]">
                Grotesque Heading & Body Stack (Inter Tight / Inter)
              </h3>
              <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">
                Serif Newsreader eliminated. Strictly 2 font weights per view (Regular 400 & Semibold 600).
              </p>
            </div>

            <div className="space-y-4 border-t border-[#E3E8EE] dark:border-[#1E2630] pt-4">
              <div>
                <span className="text-xs font-mono text-muted">Heading 1 (28px Semibold normal tracking)</span>
                <div className="text-xl font-semibold text-text tracking-normal">
                  Decide in milliseconds whether a payment should go through.
                </div>
              </div>

              <div>
                <span className="text-xs font-mono text-muted">Heading 2 (20px Semibold normal tracking)</span>
                <div className="text-lg font-semibold text-text tracking-normal">
                  Pre-authorization scoring & explainable policy routing
                </div>
              </div>

              <div>
                <span className="text-xs font-mono text-[#52606D] dark:text-[#8A96A3]">Body UI (13px Regular leading-relaxed)</span>
                <div className="text-xs text-[#52606D] dark:text-[#8A96A3] leading-relaxed max-w-2xl">
                  Evaluated strictly on public academic benchmarks and behavioral simulators across cards, wallets, accounts, and UPI. All metrics derived on held-out test splits.
                </div>
              </div>

              <div>
                <span className="text-xs font-mono text-[#52606D] dark:text-[#8A96A3]">Numbers & Alphanumerics (JetBrains Mono Tabular)</span>
                <div className="text-lg font-mono tabular-nums font-semibold text-[#0F172A] dark:text-[#E8EDF2]">
                  ₹1,248,500.00 &nbsp;·&nbsp; Calibrated Thresholds &nbsp;·&nbsp; Evaluated Test Split
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* Tab 3: Restyled Primitives */}
        <TabsContent value="primitives" className="pt-6 space-y-6">
          {/* Buttons */}
          <div className="p-5 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-3">
            <h3 className="font-semibold text-xs text-muted">
              Buttons (120ms Duration, Press Scale 0.98, Visible Focus Ring)
            </h3>
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="primary" size="md">
                <Sparkles className="w-3.5 h-3.5" strokeWidth={1.5} />
                <span>Primary Emerald</span>
              </Button>
              <Button variant="secondary" size="md">
                <span>Secondary Surface</span>
              </Button>
              <Button variant="outline" size="md">
                <span>Outline</span>
              </Button>
              <Button variant="ghost" size="md">
                <span>Ghost</span>
              </Button>
              <Button variant="destructive" size="md">
                <span>Destructive High-Risk</span>
              </Button>
            </div>
          </div>

          {/* Badges */}
          <div className="p-5 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-3">
            <h3 className="font-semibold text-xs text-muted">
              Risk & Status Badges (Amber and Red strictly reserved for risk)
            </h3>
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant="low">Low Risk (Auto-Approve)</Badge>
              <Badge variant="medium">Medium Risk (Step-Up 2FA)</Badge>
              <Badge variant="high">High Risk (Block & Alert)</Badge>
              <Badge variant="accent">Active Champion</Badge>
              <Badge variant="neutral">Neutral System Tag</Badge>
            </div>
          </div>

          {/* Form Inputs & Select */}
          <div className="p-5 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-4">
            <h3 className="font-semibold text-xs text-muted">
              Form Controls (Clean Sentence-Case Labels, Focus Rings)
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <Input label="Transaction Amount (INR)" placeholder="e.g. 45000" isMono defaultValue="45000" />
              <Input label="Sender Account / VPA" placeholder="e.g. user_9821@upi" defaultValue="user_9821@upi" />
              <Select
                label="Payment Channel"
                options={[
                  { value: 'upi', label: 'UPI India' },
                  { value: 'card', label: 'Credit Card (ULB)' },
                  { value: 'transfer_wallet', label: 'Wallets & Transfers (PaySim)' },
                  { value: 'account', label: 'Bank Account Opening (BAF)' },
                ]}
              />
            </div>
          </div>

          {/* Interactive Overlays: Sonner, Vaul Drawer, Radix Dialog */}
          <div className="p-5 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] space-y-4">
            <h3 className="font-semibold text-xs text-muted">
              Interactive Overlays (Sonner Toasts, Vaul Slide Drawer, Radix Dialog)
            </h3>
            <div className="flex flex-wrap items-center gap-3">
              <Button
                variant="secondary"
                size="sm"
                onClick={() =>
                  toast({
                    title: 'Decision Processed',
                    description: 'Transaction scored: Low Risk (Auto-Approve).',
                    variant: 'success',
                  })
                }
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" strokeWidth={1.5} />
                <span>Trigger Sonner Toast</span>
              </Button>

              <Button variant="secondary" size="sm" onClick={() => setDrawerOpen(true)}>
                <Layers className="w-3.5 h-3.5 text-emerald-500" strokeWidth={1.5} />
                <span>Open Vaul Row Drawer</span>
              </Button>

              <Button variant="secondary" size="sm" onClick={() => setDialogOpen(true)}>
                <Shield className="w-3.5 h-3.5 text-emerald-500" strokeWidth={1.5} />
                <span>Open Radix Dialog</span>
              </Button>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* Demo Dialog */}
      <Dialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        title="Radix Dialog Primitive"
        description="Clean, accessible modal with backdrop blur and keyboard Esc dismissal."
      >
        <div className="space-y-3 pt-2 text-xs">
          <p className="text-[#52606D] dark:text-[#8A96A3]">
            This dialog uses the control room dark/light palette tokens with smooth 200ms motion and visible focus management.
          </p>
          <div className="flex justify-end pt-3 border-t border-[#E3E8EE] dark:border-[#1E2630]">
            <Button variant="primary" size="sm" onClick={() => setDialogOpen(false)}>
              Understood
            </Button>
          </div>
        </div>
      </Dialog>

      {/* Demo Vaul Drawer */}
      <Drawer
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        title="Transaction TX-SAMPLE-01"
        description="Vaul slide-over row detail drawer (320ms ease-control motion)."
      >
        <div className="space-y-4 text-xs font-sans">
          <div className="p-3 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-slate-50 dark:bg-[#0B0F14] space-y-1.5 font-mono">
            <div className="flex justify-between">
              <span className="text-[#52606D] dark:text-[#8A96A3]">Calibrated Risk Score</span>
              <span className="font-semibold text-rose-500">0.942 (High Risk)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#52606D] dark:text-[#8A96A3]">Measured Latency</span>
              <span className="text-[#0F172A] dark:text-[#E8EDF2]">&lt; 15 ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#52606D] dark:text-[#8A96A3]">Policy Routing</span>
              <span className="text-rose-500">Block & Alert</span>
            </div>
          </div>

          <div className="space-y-2">
            <span className="font-semibold text-xs text-[#0F172A] dark:text-[#E8EDF2]">TreeSHAP Factor Attributions</span>
            <div className="p-2.5 rounded border border-[#E3E8EE] dark:border-[#1E2630] space-y-1">
              <div className="flex justify-between text-xs">
                <span>Payer Baseline Deviation</span>
                <span className="font-mono text-rose-500">+0.38</span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-slate-200 dark:bg-[#1E2630] overflow-hidden">
                <div className="h-full bg-rose-500" style={{ width: '75%' }} />
              </div>
            </div>
          </div>

          <div className="pt-4 flex justify-end">
            <Button variant="secondary" size="sm" onClick={() => setDrawerOpen(false)}>
              Close Drawer
            </Button>
          </div>
        </div>
      </Drawer>
    </div>
  );
};
