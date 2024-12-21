namespace osuVendetta.CLI.Menu.Pages;

public class MenuPageResponse
{
    public required MenuPageResponseType ResponseType { get; set; }
    public Type? NextMenuPageType { get; set; }

    public static MenuPageResponse ToMainMenu()
    {
        return new MenuPageResponse
        {
            ResponseType = MenuPageResponseType.ToMainMenu
        };
    }
    public static MenuPageResponse Retry()
    {
        return new MenuPageResponse
        {
            ResponseType = MenuPageResponseType.Retry
        };
    }
    public static MenuPageResponse PreviousPage()
    {
        return new MenuPageResponse
        {
            ResponseType = MenuPageResponseType.PreviousPage
        };
    }
    public static MenuPageResponse Exit()
    {
        return new MenuPageResponse
        {
            ResponseType = MenuPageResponseType.Exit
        };
    }
    public static MenuPageResponse NextPage(Type nextPageType)
    {
        return new MenuPageResponse
        {
            ResponseType = MenuPageResponseType.NextPage,
            NextMenuPageType = nextPageType,
        };
    }
    public static MenuPageResponse NextPage<TNextMenuPageType>()
        where TNextMenuPageType : MenuPage
    {
        return NextPage(typeof(TNextMenuPageType));
    }
}
