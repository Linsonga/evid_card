/**
 * 微信公众号配置示例
 * 请复制此文件并重命名为实际的配置文件，然后填入真实的AppID和AppSecret
 */

export const wechatConfig = {
  // 微信公众号AppID（请替换为您的真实AppID）
  appId: 'wx9ae26e5712133db4',
  
  // 微信公众号AppSecret（请替换为您的真实AppSecret）
  appSecret: '1c46ad2fd4a3d44073b509e303353be1'
};

// 使用示例：
import { wechatConfig } from './examples/wechat-config.js';
const { appId, appSecret } = wechatConfig;