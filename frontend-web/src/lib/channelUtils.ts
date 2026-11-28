export function getChannelCurrency(channel: string): string {
  switch (channel) {
    case 'card':
      return '€';
    case 'account':
      return '£';
    case 'transfer_wallet':
    case 'upi':
    default:
      return '₹';
  }
}

export function getChannelReadableName(channel: string): string {
  switch (channel) {
    case 'upi':
      return 'UPI Instant';
    case 'card':
      return 'Card Payment';
    case 'transfer_wallet':
      return 'Wallet Transfer';
    case 'account':
      return 'Account Opening';
    default:
      return channel;
  }
}

export function formatChannelAmount(amount: number, channel: string): string {
  const sym = getChannelCurrency(channel);
  if (channel === 'card') {
    return `${sym}${amount.toLocaleString('de-DE', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  if (channel === 'account') {
    return `${sym}${amount.toLocaleString('en-GB', { maximumFractionDigits: 0 })}`;
  }
  return `${sym}${amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}

