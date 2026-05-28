type Props = { reason: string };

export function Asterisk({ reason }: Props) {
  return (
    <span className="asterisk" title={reason}>
      *
    </span>
  );
}
